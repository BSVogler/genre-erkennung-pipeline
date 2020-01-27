#!python3

from http.server import HTTPServer, CGIHTTPRequestHandler
import webbrowser
import threading
import os
import sys
import time

def start_server(path, port=8000):
    '''Start a simple webserver serving path on port'''
    os.chdir(path)
    httpd = HTTPServer(('', port), CGIHTTPRequestHandler)
    httpd.serve_forever()

# Start the server in a new thread
port = 8000
daemon = threading.Thread(name='daemon_server',
                          target=start_server,
                          args=('.', port))
daemon.setDaemon(True)
daemon.start()

# Open the web browser 
webbrowser.open('http://localhost:{}'.format(port))

while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        sys.exit(0)