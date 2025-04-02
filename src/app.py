
from flask import Flask
from skewt.routes import skewt_bp

app = Flask(__name__)

app.register_blueprint(skewt_bp)

if __name__ == "__main__":
    # Make sure to allow inbound traffic on the port in your EC2 Security Group
    app.run(host="0.0.0.0", port=5000, debug=True)
