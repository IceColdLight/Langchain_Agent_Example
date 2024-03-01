import streamlit as st
from streamlit import config as _config
from streamlit.web.bootstrap import run
import os

class Web_Interface:
    def __init__(self):
        self.setup_server()
    
    def setup_server(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'ui.py')

        _config.set_option("server.headless", True)
        run(filename, args=[], flag_options=[], is_hello=False)