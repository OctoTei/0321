#!/usr/bin/env python3
# socket_communication.py - Socket-based communication for distributed FedTGP

import logging
import pickle
import socket
import time

from checkpoint_manager import CheckpointManager


_HEADER_SIZE = 4
_RECV_CHUNK_SIZE = 4096


def _recv_exactly(sock, num_bytes, peer_name="peer"):
    """
    Read exactly num_bytes from a TCP socket.
    This is required because sock.recv(n) may return fewer than n bytes.
    """
    if num_bytes <= 0:
        return b""

    data = bytearray()
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            if len(data) == 0:
                raise ConnectionError(f"{peer_name} disconnected - no data received")
            raise ConnectionError(
                f"{peer_name} disconnected during receive "
                f"({len(data)}/{num_bytes} bytes received)"
            )
        data.extend(chunk)
    return bytes(data)


class FedTGPSocketServer:
    """Socket-based FedTGP Server for cross-device communication with reconnection support"""

    def __init__(self, args, aggregator, port, num_clients):
        self.args = args
        self.aggregator = aggregator
        self.port = port
        self.num_clients = num_clients
        self.client_sockets = {}
        self.max_rounds = args.comm_round

        self.checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")
        self.checkpoint_interval = 5

        self.max_reconnect_attempts = 10
        self.reconnect_wait_time = 5

        self.initial_connect_timeout = getattr(args, "initial_connect_timeout", 300)
        self.accept_poll_interval = getattr(args, "accept_poll_interval", 5)
        self.client_handshake_timeout = getattr(args, "client_handshake_timeout", 15)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        logging.info(f"FedTGP Socket Server initialized on port {port} with reconnection support")

    def send_message(self, client_socket, message):
        try:
            data = pickle.dumps(message)
            message_size = len(data)
            client_socket.sendall(message_size.to_bytes(_HEADER_SIZE, byteorder="big"))
            client_socket.sendall(data)
            return True
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            raise

    def receive_message(self, client_socket):
        try:
            size_data = _recv_exactly(client_socket, _HEADER_SIZE, peer_name="Peer")
            message_size = int.from_bytes(size_data, byteorder="big")

            data = bytearray()
            while len(data) < message_size:
                chunk = client_socket.recv(min(message_size - len(data), _RECV_CHUNK_SIZE))
                if not chunk:
                    raise ConnectionError(
                        f"Peer disconnected during message receive "
                        f"({len(data)}/{message_size} bytes received)"
                    )
                data.extend(chunk)

            return pickle.loads(bytes(data))
        except Exception as e:
            logging.error(f"Error receiving message: {e}")
            raise

    def _close_socket_quietly(self, sock):
        if sock is None:
            return
        try:
            sock.close()
        except Exception:
            pass

    def _accept_and_register_client(self, expected_client_id=None, timeout=None):
        previous_timeout = self.server_socket.gettimeout()
        try:
            self.server_socket.settimeout(timeout)
            client_socket, client_address = self.server_socket.accept()
            client_socket.settimeout(self.client_handshake_timeout)

            hello = self.receive_message(client_socket)
            if not hello or hello.get("type") != "hello":
                self._close_socket_quietly(client_socket)
                raise ValueError(f"Invalid handshake from {client_address}: {hello}")

            requested_client_id = int(hello.get("requested_client_id"))
            if requested_client_id < 1 or requested_client_id > self.num_clients:
                self._close_socket_quietly(client_socket)
                raise ValueError(f"Invalid requested_client_id={requested_client_id} from {client_address}")

            if expected_client_id is not None and requested_client_id != expected_client_id:
                self._close_socket_quietly(client_socket)
                raise ValueError(
                    f"Reconnection client_id mismatch: expected {expected_client_id}, got {requested_client_id}"
                )

            if expected_client_id is None and requested_client_id in self.client_sockets:
                self._close_socket_quietly(client_socket)
                raise ValueError(f"Duplicate initial connection for client {requested_client_id}")

            client_socket.settimeout(None)
            self.client_sockets[requested_client_id] = client_socket
            self.send_message(client_socket, {"type": "client_id", "client_id": requested_client_id})
            return requested_client_id, client_address
        finally:
            self.server_socket.settimeout(previous_timeout)

    def start_server(self):
        try:
            self.server_socket.bind(("0.0.0.0", self.port))
            self.server_socket.listen(self.num_clients)
            logging.info(f"FedTGP Server listening on 0.0.0.0:{self.port}")
            print(f"✓ FedTGP Server listening on 0.0.0.0:{self.port}")
            print("✓ Server is ready to accept connections")

            logging.info(
                f"Waiting for {self.num_clients} clients to connect "
                f"(timeout={self.initial_connect_timeout}s, poll={self.accept_poll_interval}s)..."
            )
            print(f"⏳ Waiting for {self.num_clients} clients to connect...")
            print(f"⏳ Initial connection timeout: {self.initial_connect_timeout}s")

            deadline = time.time() + max(float(self.initial_connect_timeout), 1.0)

            while len(self.client_sockets) < self.num_clients:
                remaining_clients = self.num_clients - len(self.client_sockets)
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    logging.error(
                        f"Initial connection timeout reached. Connected "
                        f"{len(self.client_sockets)}/{self.num_clients} clients."
                    )
                    print(f"\n{'=' * 60}")
                    print("✗ Initial connection timeout reached")
                    print(f"✗ Connected {len(self.client_sockets)}/{self.num_clients} clients")
                    print("✗ Server will terminate instead of waiting forever")
                    print(f"{'=' * 60}")
                    return False

                current_timeout = min(float(self.accept_poll_interval), remaining_time)
                print(
                    f"⏳ Waiting for {remaining_clients} more client(s)... "
                    f"(remaining {remaining_time:.1f}s)"
                )
                try:
                    client_id, client_address = self._accept_and_register_client(
                        expected_client_id=None,
                        timeout=current_timeout,
                    )
                    logging.info(f"Client {client_id} connected from {client_address}")
                    print(f"✓ Client {client_id} connected from {client_address}")
                except socket.timeout:
                    logging.info(
                        f"Initial accept timed out after {current_timeout:.1f}s; "
                        f"still waiting for remaining clients"
                    )
                    continue
                except Exception as e:
                    logging.error(f"Error accepting client: {e}")
                    print(f"✗ Error accepting client: {e}")

            logging.info(f"All {self.num_clients} clients connected successfully")
            print(f"✓ All {self.num_clients} clients connected successfully")
            print("=" * 60)
            return True
        except Exception as e:
            logging.error(f"Error starting server: {e}")
            print(f"✗ Error starting server: {e}")
            import traceback
            traceback.print_exc()
            return False

    def wait_for_client_reconnection(self, client_id):
        logging.info(f"Waiting for client {client_id} to reconnect...")
        print(f"\n{'=' * 60}")
        print(f"⚠ Client {client_id} disconnected - waiting for reconnection")
        print(f"{'=' * 60}")

        old_socket = self.client_sockets.get(client_id)
        self._close_socket_quietly(old_socket)
        if client_id in self.client_sockets:
            del self.client_sockets[client_id]

        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                logging.info(f"Reconnection attempt {attempt}/{self.max_reconnect_attempts} for client {client_id}")
                print(f"⏳ Reconnection attempt {attempt}/{self.max_reconnect_attempts} - waiting {self.reconnect_wait_time}s...")
                reconnected_id, client_address = self._accept_and_register_client(
                    expected_client_id=client_id,
                    timeout=self.reconnect_wait_time,
                )
                logging.info(f"✓ Client {reconnected_id} reconnected from {client_address}")
                print(f"✓ Client {reconnected_id} reconnected successfully from {client_address}")
                print(f"{'=' * 60}\n")
                return True
            except socket.timeout:
                logging.warning(
                    f"Reconnection timeout for client {client_id} "
                    f"(attempt {attempt}/{self.max_reconnect_attempts})"
                )
                continue
            except Exception as e:
                logging.error(f"Error during reconnection attempt {attempt}: {e}")
                continue

        logging.error(f"✗ Client {client_id} failed to reconnect after {self.max_reconnect_attempts} attempts")
        print(f"\n{'=' * 60}")
        print(f"✗ Client {client_id} failed to reconnect")
        print("✗ Training will be terminated")
        print(f"{'=' * 60}\n")
        return False

    def broadcast_message(self, message):
        max_broadcast_attempts = 3
        for client_id in sorted(list(self.client_sockets.keys())):
            success = False
            for attempt in range(max_broadcast_attempts):
                try:
                    self.send_message(self.client_sockets[client_id], message)
                    logging.debug(f"Successfully sent message to client {client_id}")
                    success = True
                    break
                except Exception as e:
                    logging.error(f"Failed to send message to client {client_id} (attempt {attempt + 1}): {e}")
                    if not self.wait_for_client_reconnection(client_id):
                        raise ConnectionError(f"Client {client_id} failed to reconnect")
            if not success:
                raise ConnectionError(f"Failed to broadcast to client {client_id} after {max_broadcast_attempts} attempts")

    def _resend_current_round_to_client(self, client_id, round_num, global_prototypes):
        reconnect_message = {
            "type": "global_prototypes",
            "round": round_num,
            "prototypes": global_prototypes,
            "resume_after_reconnect": True,
        }
        self.send_message(self.client_sockets[client_id], reconnect_message)
        logging.info(f"Re-sent current round {round_num + 1} global prototypes to client {client_id}")
        print(f"✓ Re-sent current round message to client {client_id}")

    def run(self):
        if not self.start_server():
            logging.error("Failed to start server")
            print("✗ Failed to start server")
            return False

        training_ok = False
        try:
            global_prototypes, start_round = self.checkpoint_manager.load_server_checkpoint(self.aggregator)
            if global_prototypes is None:
                global_prototypes = {}
                start_round = 0

            print(f"\n{'=' * 60}")
            print(f"Starting from round index {start_round} (next displayed round: {start_round + 1})")
            print(f"{'=' * 60}\n")

            for round_num in range(start_round, self.max_rounds):
                logging.info(f"=== FedTGP Round {round_num + 1}/{self.max_rounds} ===")
                print(f"\n{'=' * 60}")
                print(f"FedTGP Round {round_num + 1}/{self.max_rounds}")
                print(f"{'=' * 60}")

                message = {
                    "type": "global_prototypes",
                    "round": round_num,
                    "prototypes": global_prototypes,
                }

                try:
                    self.broadcast_message(message)
                    if global_prototypes:
                        logging.info(f"FedTGP Server: Sent {len(global_prototypes)} TGP-generated global prototypes to all clients")
                        print(f"✓ Sent {len(global_prototypes)} global prototypes to all clients")
                    else:
                        logging.info("FedTGP Server: Sent empty global prototypes (first round)")
                        print("✓ Sent empty global prototypes to all clients (first round)")
                except ConnectionError as e:
                    logging.error(f"Failed to broadcast to clients: {e}")
                    print(f"\n{'=' * 60}")
                    print("✗ Training stopped: Client disconnected during broadcast")
                    print(f"✗ Error: {e}")
                    print(f"{'=' * 60}")
                    return False

                client_prototypes = {}
                print(f"⏳ Waiting for {self.num_clients} clients to send prototypes...")

                for client_id in range(1, self.num_clients + 1):
                    received = False
                    max_receive_attempts = 3

                    for attempt in range(max_receive_attempts):
                        try:
                            response = self.receive_message(self.client_sockets[client_id])
                            if response and response.get("type") == "client_prototypes":
                                response_client_id = int(response.get("client_id", -1))
                                response_round = int(response.get("round", -1))
                                if response_client_id != client_id:
                                    raise ValueError(
                                        f"client_id mismatch: socket slot={client_id}, payload={response_client_id}"
                                    )
                                if response_round != round_num:
                                    raise ValueError(
                                        f"round mismatch from client {client_id}: expected {round_num}, got {response_round}"
                                    )

                                client_prototypes[client_id] = response["prototypes"]
                                logging.info(f"Received {len(response['prototypes'])} prototypes from client {client_id}")
                                print(f"✓ Received {len(response['prototypes'])} prototypes from client {client_id}")
                                received = True
                                break

                            raise ValueError(f"Invalid response from client {client_id}: {response}")

                        except Exception as e:
                            logging.error(f"Failed to receive prototypes from client {client_id} (attempt {attempt + 1}): {e}")
                            print(f"✗ Failed to receive from client {client_id} (attempt {attempt + 1}): {e}")

                            if not self.wait_for_client_reconnection(client_id):
                                logging.error(f"Client {client_id} failed to reconnect, stopping training")
                                print(f"\n{'=' * 60}")
                                print(f"✗ Training stopped: Client {client_id} failed to reconnect")
                                print(f"{'=' * 60}")
                                return False

                            print(f"⏳ Re-synchronizing client {client_id} with current round {round_num + 1}...")
                            try:
                                self._resend_current_round_to_client(client_id, round_num, global_prototypes)
                            except Exception as resend_error:
                                logging.error(f"Failed to re-send current round to client {client_id}: {resend_error}")
                                print(f"✗ Failed to re-send current round to client {client_id}: {resend_error}")

                    if not received:
                        logging.error(f"Failed to receive from client {client_id} after {max_receive_attempts} attempts")
                        print(f"\n{'=' * 60}")
                        print(f"✗ Training stopped: Failed to receive from client {client_id}")
                        print(f"{'=' * 60}")
                        return False

                if not client_prototypes:
                    logging.error("No client prototypes received, stopping training")
                    print("✗ No client prototypes received, stopping training")
                    return False

                print("⏳ Aggregating prototypes using TGP...")
                global_prototypes = self.aggregator.aggregate_prototypes(client_prototypes)
                logging.info(f"TGP generated {len(global_prototypes)} global prototypes")
                print(f"✓ TGP generated {len(global_prototypes)} global prototypes")
                print(f"✓ Round {round_num + 1} completed")
                logging.info(f"Round {round_num + 1} completed")

                if (round_num + 1) % self.checkpoint_interval == 0:
                    self.checkpoint_manager.save_server_checkpoint(round_num + 1, self.aggregator, global_prototypes)
                    self.checkpoint_manager.cleanup_old_checkpoints("server", keep_last_n=3)

            try:
                self.broadcast_message({"type": "training_finished"})
                logging.info("FedTGP training completed")
                print(f"\n{'=' * 60}")
                print("✓ FedTGP training completed successfully")
                print(f"{'=' * 60}")
            except ConnectionError as e:
                logging.warning(f"Failed to send training_finished signal: {e}")
                print(f"\n{'=' * 60}")
                print("⚠ Training completed but some clients may have disconnected")
                print(f"{'=' * 60}")

            training_ok = True
            return True
        except Exception as e:
            logging.error(f"Error in server run: {e}")
            print(f"✗ Error in server run: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()
            logging.info(f"Server run finished with status={training_ok}")

    def cleanup(self):
        for client_socket in self.client_sockets.values():
            self._close_socket_quietly(client_socket)
        self.client_sockets.clear()
        self._close_socket_quietly(self.server_socket)
        logging.info("Server cleanup completed")


class FedTGPSocketClient:
    """Socket-based FedTGP Client for cross-device communication with auto-reconnection"""

    def __init__(self, args, trainer, server_ip, server_port, client_id):
        self.args = args
        self.trainer = trainer
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_id = client_id
        self.socket = None

        self.checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")
        self.checkpoint_interval = 5

        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        self.resync_timeout = 30

        logging.info(f"FedTGP Socket Client {client_id} initialized with auto-reconnection")

    def _close_socket_quietly(self):
        if self.socket is None:
            return
        try:
            self.socket.close()
        except Exception:
            pass
        self.socket = None

    def _perform_handshake(self):
        hello = {
            "type": "hello",
            "requested_client_id": self.client_id,
        }
        self.send_message(hello)
        response = self.receive_message()
        if not response or response.get("type") != "client_id":
            raise ConnectionError(f"Invalid handshake response from server: {response}")
        assigned_id = int(response["client_id"])
        if assigned_id != self.client_id:
            raise ConnectionError(
                f"Server assigned different ID: {assigned_id} (expected {self.client_id})"
            )
        logging.info(f"Server confirmed client ID: {assigned_id}")
        print(f"✓ Server confirmed client ID: {assigned_id}")

    def reconnect_to_server(self):
        logging.info(f"Client {self.client_id}: Attempting to reconnect to server...")
        print(f"\n{'=' * 60}")
        print("⚠ Connection lost - attempting to reconnect")
        print(f"{'=' * 60}")

        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                logging.info(f"Reconnection attempt {attempt}/{self.max_reconnect_attempts}")
                print(f"⏳ Reconnection attempt {attempt}/{self.max_reconnect_attempts}...")

                self._close_socket_quietly()
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10)
                self.socket.connect((self.server_ip, self.server_port))
                self._perform_handshake()
                self.socket.settimeout(None)
                logging.info(f"✓ Client {self.client_id} reconnected successfully")
                print("✓ Reconnected successfully to server")
                print(f"{'=' * 60}\n")
                return True
            except Exception as e:
                logging.error(f"Reconnection attempt {attempt} failed: {e}")
                print(f"✗ Reconnection attempt {attempt} failed: {e}")
                self._close_socket_quietly()
                if attempt < self.max_reconnect_attempts:
                    print(f"⏳ Waiting {self.reconnect_delay}s before next attempt...")
                    time.sleep(self.reconnect_delay)

        logging.error(f"✗ Client {self.client_id} failed to reconnect after {self.max_reconnect_attempts} attempts")
        print(f"\n{'=' * 60}")
        print("✗ Failed to reconnect to server")
        print("✗ Client will terminate")
        print(f"{'=' * 60}\n")
        return False

    def connect_to_server(self):
        max_retries = 5
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                logging.info(f"Connecting to server {self.server_ip}:{self.server_port} (attempt {attempt + 1}/{max_retries})")
                print(f"⏳ Connecting to server {self.server_ip}:{self.server_port} (attempt {attempt + 1}/{max_retries})...")

                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10)
                self.socket.connect((self.server_ip, self.server_port))

                logging.info(f"Connected to server {self.server_ip}:{self.server_port}")
                print(f"✓ Connected to server {self.server_ip}:{self.server_port}")

                self._perform_handshake()
                self.socket.settimeout(None)
                logging.info("Client socket timeout removed - ready to wait for server messages")
                return True
            except socket.timeout:
                logging.warning(f"Connection timeout (attempt {attempt + 1}/{max_retries})")
                print(f"✗ Connection timeout (attempt {attempt + 1}/{max_retries})")
                self._close_socket_quietly()
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            except ConnectionRefusedError:
                logging.warning(f"Connection refused (attempt {attempt + 1}/{max_retries})")
                print(f"✗ Connection refused - is the server running? (attempt {attempt + 1}/{max_retries})")
                self._close_socket_quietly()
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            except Exception as e:
                logging.error(f"Error connecting to server (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"✗ Error connecting to server: {e}")
                self._close_socket_quietly()
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        logging.error(f"Failed to connect to server after {max_retries} attempts")
        print(f"✗ Failed to connect to server after {max_retries} attempts")
        return False

    def send_message(self, message):
        try:
            data = pickle.dumps(message)
            message_size = len(data)
            self.socket.sendall(message_size.to_bytes(_HEADER_SIZE, byteorder="big"))
            self.socket.sendall(data)
            return True
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            raise ConnectionError(f"Send error: {e}")

    def receive_message(self):
        try:
            size_data = _recv_exactly(self.socket, _HEADER_SIZE, peer_name="Server")
            message_size = int.from_bytes(size_data, byteorder="big")

            data = bytearray()
            while len(data) < message_size:
                chunk = self.socket.recv(min(message_size - len(data), _RECV_CHUNK_SIZE))
                if not chunk:
                    raise ConnectionError(
                        f"Server disconnected during message receive "
                        f"({len(data)}/{message_size} bytes received)"
                    )
                data.extend(chunk)

            return pickle.loads(bytes(data))
        except ConnectionError:
            raise
        except Exception as e:
            logging.error(f"Error receiving message: {e}")
            raise ConnectionError(f"Receive error: {e}")

    def _sync_current_round_after_reconnect(self, expected_round):
        """
        After reconnecting during the prototype-send phase, the server re-sends the
        current round's global_prototypes so the client and server stay aligned.
        Consume that message before re-sending cached prototypes.
        """
        previous_timeout = self.socket.gettimeout()
        try:
            self.socket.settimeout(self.resync_timeout)
            message = self.receive_message()
        finally:
            self.socket.settimeout(previous_timeout)

        if not message:
            raise ConnectionError("Empty resynchronization message from server")

        message_type = message.get("type")
        if message_type == "training_finished":
            raise ConnectionError("Server reported training_finished during reconnection")

        if message_type != "global_prototypes":
            raise ConnectionError(
                f"Unexpected message during reconnection sync: {message_type}"
            )

        resent_round = int(message.get("round", -1))
        if resent_round != expected_round:
            raise ConnectionError(
                f"Resynchronization round mismatch: expected {expected_round}, got {resent_round}"
            )

        resent_prototypes = message.get("prototypes", {})
        if resent_prototypes:
            self.trainer.update_global_prototypes(resent_prototypes)
            logging.info(
                f"Client {self.client_id}: Re-synchronized global prototypes for round {resent_round + 1}"
            )
        else:
            logging.info(
                f"Client {self.client_id}: Re-synchronized empty global prototypes for round {resent_round + 1}"
            )

        print(f"✓ Re-synchronized with server for round {resent_round + 1}")
        return True

    def run(self):
        if not self.connect_to_server():
            logging.error("Failed to connect to server")
            print("✗ Failed to connect to server")
            return False

        print("=" * 60)
        print(f"✓ Client {self.client_id} ready for training")
        print("=" * 60)

        run_ok = False
        last_completed_round = -1
        last_round_local_prototypes = None

        try:
            _, start_round = self.checkpoint_manager.load_client_checkpoint(
                self.client_id, self.trainer.model, self.trainer.device
            )
            print(f"\n{'=' * 60}")
            print(f"Starting from local checkpoint round index {start_round}")
            print(f"{'=' * 60}\n")

            while True:
                message = None
                max_attempts = 3

                for attempt in range(max_attempts):
                    try:
                        message = self.receive_message()
                        break
                    except ConnectionError as e:
                        logging.error(f"Connection error (attempt {attempt + 1}): {e}")
                        print(f"✗ Connection error: {e}")
                        if not self.reconnect_to_server():
                            print(f"\n{'=' * 60}")
                            print(f"✗ Client {self.client_id} terminating - cannot reconnect")
                            print(f"{'=' * 60}")
                            return False
                        print("⏳ Retrying to receive message...")

                if not message:
                    logging.error("Failed to receive message after reconnection attempts")
                    print("✗ Failed to receive message from server")
                    return False

                if message["type"] == "global_prototypes":
                    global_prototypes = message["prototypes"]
                    round_num = int(message["round"])
                    self.trainer.current_round = round_num

                    if round_num == last_completed_round:
                        logging.warning(
                            f"Client {self.client_id}: duplicate control message for round {round_num + 1}; "
                            f"re-sending cached prototypes instead of retraining"
                        )
                        print(f"⚠ Duplicate round {round_num + 1} message received, re-sending cached prototypes")
                        if last_round_local_prototypes is None:
                            logging.error("No cached local prototypes available for duplicate round")
                            print("✗ No cached local prototypes available for duplicate round")
                            return False

                        duplicate_response = {
                            "type": "client_prototypes",
                            "client_id": self.client_id,
                            "round": round_num,
                            "prototypes": last_round_local_prototypes,
                        }
                        self.send_message(duplicate_response)
                        print(f"✓ Re-sent cached {len(last_round_local_prototypes)} local prototypes to server")
                        continue

                    if round_num < last_completed_round:
                        logging.error(
                            f"Client {self.client_id}: stale round message received. "
                            f"last_completed_round={last_completed_round}, received={round_num}"
                        )
                        print(f"✗ Stale round message received: {round_num + 1}")
                        return False

                    logging.info(
                        f"Client {self.client_id}: Round {round_num + 1}, "
                        f"received {len(global_prototypes)} global prototypes"
                    )
                    print(f"\n{'=' * 60}")
                    print(f"Round {round_num + 1}")
                    print(f"{'=' * 60}")
                    print(f"✓ Received {len(global_prototypes)} global prototypes")

                    if global_prototypes:
                        self.trainer.update_global_prototypes(global_prototypes)
                        logging.info(f"Client {self.client_id}: Updated global prototypes")
                        print("✓ Updated global prototypes")
                    else:
                        logging.info(f"Client {self.client_id}: No global prototypes (first round)")
                        print("✓ First round - no global prototypes yet")

                    try:
                        print("⏳ Training local model...")
                        local_prototypes = self.trainer.train_and_extract_prototypes()
                        print(f"✓ Training completed, extracted {len(local_prototypes)} local prototypes")
                        if not local_prototypes:
                            logging.error("No local prototypes extracted, cannot continue")
                            print("✗ No local prototypes extracted")
                            return False
                    except Exception as train_error:
                        logging.error(f"Training failed: {train_error}")
                        print(f"✗ Training failed: {train_error}")
                        import traceback
                        traceback.print_exc()
                        return False

                    response = {
                        "type": "client_prototypes",
                        "client_id": self.client_id,
                        "round": round_num,
                        "prototypes": local_prototypes,
                    }

                    sent = False
                    for attempt in range(max_attempts):
                        try:
                            self.send_message(response)
                            logging.info(
                                f"FedTGP Client {self.client_id}: Sent {len(local_prototypes)} local prototypes to server"
                            )
                            print(f"✓ Sent {len(local_prototypes)} local prototypes to server")
                            sent = True
                            last_completed_round = round_num
                            last_round_local_prototypes = local_prototypes
                            break
                        except ConnectionError as e:
                            logging.error(f"Failed to send prototypes (attempt {attempt + 1}): {e}")
                            print(f"✗ Failed to send prototypes: {e}")
                            if not self.reconnect_to_server():
                                print(f"\n{'=' * 60}")
                                print(f"✗ Client {self.client_id} terminating - cannot reconnect")
                                print(f"{'=' * 60}")
                                return False

                            try:
                                self._sync_current_round_after_reconnect(expected_round=round_num)
                            except ConnectionError as sync_error:
                                logging.error(f"Failed to re-synchronize after reconnect: {sync_error}")
                                print(f"✗ Failed to re-synchronize after reconnect: {sync_error}")
                                continue

                            print("⏳ Reconnected and synchronized, retrying to send prototypes...")

                    if not sent:
                        logging.error("Failed to send prototypes after reconnection attempts")
                        print("✗ Failed to send prototypes to server")
                        return False

                    if (round_num + 1) % self.checkpoint_interval == 0:
                        self.checkpoint_manager.save_client_checkpoint(
                            self.client_id, round_num + 1, self.trainer.model, local_prototypes
                        )
                        self.checkpoint_manager.cleanup_old_checkpoints(
                            "client", entity_id=self.client_id, keep_last_n=3
                        )

                elif message["type"] == "training_finished":
                    logging.info("Training finished signal received")
                    print(f"\n{'=' * 60}")
                    print("✓ Training finished successfully")
                    print(f"{'=' * 60}")
                    run_ok = True
                    break
                else:
                    logging.error(f"Unknown message type from server: {message.get('type')}")
                    print(f"✗ Unknown message type from server: {message.get('type')}")
                    return False

            return run_ok
        except Exception as e:
            logging.error(f"Error in client run: {e}")
            print(f"✗ Error in client run: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()
            logging.info(f"Client run finished with status={run_ok}")

    def cleanup(self):
        self._close_socket_quietly()
        logging.info("Client cleanup completed")
