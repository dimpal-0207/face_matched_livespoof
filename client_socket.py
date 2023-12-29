import socketio

sio = socketio.Client()
print("===sio", sio)


@sio.event
def message(data):
    print('I received a message!')

@sio.on('message')
def on_message(data):
    print('I received a message!')

if __name__ == '__main__':
    sio.connect('http://127.0.0.1:5001')
    sio.wait()
