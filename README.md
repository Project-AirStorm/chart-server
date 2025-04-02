# Testing 
### Run Test Server for local SVG debugging from metpy 
python3 -m http.server 8000

### Delete all from S3 bucket
`aws s3 rm s3://meteo-charts/skewt-diagrams --recursive`

### Download all SVGs from a specific folder (local testing)
`aws s3 sync "s3://meteo-charts/skewt-diagrams/skewt-dumps_user_2sirXuIdmQh7eiB3GwHxZlcQYbI/chart_2025-03-11_12-41-18/" test/svg-dumps/`   
`aws s3 sync "s3://meteo-charts/skewt-diagrams/skewt-dumps_user_2seeKmUaxI6vzlvi1jzLguWFQZ8/chart_2025-03-11_22-11-15/" test/svg-dumps/`

### Test EC2 endpoint WITHIN EC2 Instance:
Generates a skew-t for one day:  
`curl "http://localhost:5000/generate-skew?days=1&lat=32.52&lon=-93.75&user_id=user_2sirXuIdmQh7eiB3GwHxZlcQYbI"`   

### Test EC2 endpoints _OUTSIDE_ EC2 Instance:
`curl -v "http://ec2-3-221-177-106.compute-1.amazonaws.com:5000/generate-skew?days=1&lat=52.537&lon=13.376&user_id=user_2sirXuIdmQh7eiB3GwHxZlcQYbI"`

`curl -v "http://ec2-3-221-177-106.compute-1.amazonaws.com:5000/generate-skew?days=1&lat=52.537&lon=13.376&user_id=user_2seeKmUaxI6vzlvi1jzLguWFQZ8"`



# Running in Production on EC2
### .flaskenv
```
FLASK_APP=app.py
FLASK_RUN_HOST=0.0.0.0
FLASK_RUN_PORT=5000
FLASK_ENV=production
```
### Gunicorn
Gunicorn enables us to keep out EC2 running with out an SSH/AWS EC2 Console open:   
`gunicorn -b 0.0.0.0:5000 app:app`


### The daemon for flask server:
Located located in `etc/systemd/system`     

`chart-server.service`:
```
[Unit]
Description=Chart Server Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/chart-server/src
ExecStart=/home/ubuntu/chart-server/.venv/bin/gunicorn -b 0.0.0.0:5000 app:app
Restart=always
Environment=FLASK_ENV=production

[Install]
WantedBy=multi-user.target
```

### Start the chart-server service on EC2:
```
sudo systemctl daemon-reload
sudo systemctl start chart-server
sudo systemctl enable chart-server
```

### Debugging:
Ensure the gunicorn server is running on port 5000:
`sudo lsof -i :5000`

Expected output:
```
COMMAND   PID   USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
gunicorn 8064 ubuntu    5u  IPv4  48835      0t0  TCP *:5000 (LISTEN)
gunicorn 8111 ubuntu    5u  IPv4  48835      0t0  TCP *:5000 (LISTEN)
```