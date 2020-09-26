considering the limited computing power of raspberry pi, we output the offloading actions to a file in advance

Future work will consider implementing an independent control center to make offloading decisions for all raspberry pis

Also, we print the messages we need to calculate reward:

including: 
* cpu occupied time, to calculate the cpu cost
* task finish time (including queue time), help us to determine is after deadline
 
In addition, as TCP sockets are in "blocking" mode, we can not directly print the time like:

```python
trans_begin = time.time()
client_socket.recv(1024)
trans_end = time.time()
``` 

As a compromise, we use a seperate program to collect the transmission rate under different situations.
