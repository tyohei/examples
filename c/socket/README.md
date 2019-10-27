---
Date: 2019-10-23
Name: Yohei Tsuji
---
# Socket API

Read CS:APP3e page 932.

```sh
make
./main 9999
# Then access to localhost:9999 from the browser
```

Based on [this article](https://qiita.com/tajima_taso/items/2f0606db7764580cf295).
Based on [CS:APP](https://csapp.cs.cmu.edu)

- socket
  - Each socket has a corresponding *socket address* and a *port*.
- connections
  - Given socker and client sockets, the connection between the server and the client is uniquely identified by the ***socket pair***.
- The *socket interface*
  - is a set of functions that are used to build network applications.

Client
- `getaddrinfo`
- `socket`
- `connect`
- `rio_written`
- `rio_readlineb`
- `close`

Server
- `getaddrinfo`
- `socket`
- `bind`
- `listen`
- `accept`
- `rio_readlineb`
- `rio_written`
- `rio_readlineb`
- `close`
