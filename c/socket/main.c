#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>


int main(int argc, char const* argv[])
{
  int server_socket;  // from the perspective of a Linux program, a socket is an open file with a corresponding descriptor.
  int connected_socket;

  struct sockaddr_in server_sockaddr;  // Internet socket addresses are stored in this 16-byte struct
  struct sockaddr_in client_sockaddr;
  /*
   * // IP socket address structure
   * // ``sin'' is an abbr of "Socketaddr IN".
   * struct sockaddr_in {
   *    uint16_t       sin_family;   // Protocol familiy (always AF_INET)
   *    uint16_t       sin_port;     // Port number in network byte order (big endian)
   *    struct in_addr sin_addr;     // IP address in network byte oder (big endian)
   *    unsigned char  sin_zero[8];  // pad to sizeof(struct sockaddr)
   * };
   *
   *
   * // Generic socket address structure (for connect, bind, and accept).
   * struct sockaddr {
   *    uint16_t sa_family;
   *    char     sa_data[14];
   * };
   */

  unsigned short server_port;

  if (argc != 2) {
    fprintf(stderr, "arg count mismatch\n");
    exit(EXIT_FAILURE);
  }

  server_port = (unsigned short)atoi(argv[1]);
  if (server_port == 0) {
    fprintf(stderr, "invalid port number\n");
    exit(EXIT_FAILURE);
  }

  server_socket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (server_socket < 0) {
    perror("socket() failed");
    exit(EXIT_FAILURE);
  }

  memset(&server_sockaddr, 0, sizeof(server_sockaddr));
  server_sockaddr.sin_family      = AF_INET;
  server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  server_sockaddr.sin_port        = htons(server_port);

  int bind_res = bind(server_socket, (struct sockaddr *)&server_sockaddr, sizeof(server_sockaddr));
  // ``bind'' associates fd and addr (here, these are server_socket and server_sockaddr).
  if (bind_res < 0) {
    perror("bind() failed");
    exit(EXIT_FAILURE);
  }

  int listen_res = listen(server_socket, 5);
  if (listen_res < 0) {
    perror("listen() failed");
    exit(EXIT_FAILURE);
  }

  unsigned int client_len;
  char read_buffer[1024*1024 + 1];
  int rc = 0;
  while (1) {
    client_len = sizeof(client_sockaddr);
    connected_socket = accept(server_socket, (struct sockaddr *)&client_sockaddr, &client_len);
    if (connected_socket < 0) {
      perror("accept() failed");
      exit(EXIT_FAILURE);
    }

    // Read the data from client and write to client.
    rc = read(connected_socket, read_buffer, 1024*1024);
    printf("read: %d - %s\n", rc, read_buffer);

    printf("connected from %s\n", inet_ntoa(client_sockaddr.sin_addr));
    close(connected_socket);
  }

  return 0;
}
