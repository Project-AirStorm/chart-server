# # app.py

# import os
# import io
# import time
# import json
# import requests
# import boto3
# from flask import Flask, request, jsonify
# from generate_skew import parse_json, plot_skewt_from_json
# # import the parse_json and plot_skewt_from_json functions from generate_skew.py

# app = Flask(__name__)
 
# s3_client = boto3.client("s3")

# # Optionally store your S3 bucket in an environment variable
# S3_BUCKET = os.environ.get("S3_BUCKET", "my-bucket-name")

# @app.route("/generate-skew", methods=["GET"])
# def generate_skew():
#     """
#     Example endpoint: /generate-skew?days=3&lat=32.52&lon=-93.75
#     """
#     # 1. Grab query params
#     days = request.args.get("days", default="1", type=int)
#     lat = request.args.get("lat", default="32.52", type=float)
#     lon = request.args.get("lon", default="-93.75", type=float)

#     # 2. Build the Open-Meteo API URL based on these params
#     #    Here, we just replicate your example with the included fields:
#     #    (Note: if you want fewer variables, you can strip them out.)
#     url = (
#         "https://api.open-meteo.com/v1/forecast"
#         f"?latitude={lat}"
#         f"&longitude={lon}"
#         # All the variables you need:
#         "&hourly=temperature_1000hPa,temperature_975hPa,temperature_950hPa,..."
#         "&forecast_days={days}"
#         "&models=gfs_graphcast025"
#     )

#     # 3. Fetch the weather data from Open-Meteo
#     try:
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
#         weather_json = response.json()
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     # 4. Loop over hours you care about
#     #    Let's say we handle all hours up to the # of hours in the forecast
#     #    or simply days*24. Adjust this logic to your needs.
#     hours_to_process = days * 24
#     all_s3_urls = []

#     for hour_index in range(hours_to_process):
#         # Parse the data for this hour
#         parsed_data = parse_json(weather_json, hour_index=hour_index)

#         # Generate the SkewT/Hodo as an in-memory SVG
#         # We'll do an in-memory BytesIO to avoid local filesystem usage if desired
#         out_filename = f"skewt_{lat}_{lon}_hour_{hour_index}.svg"
#         svg_buffer = io.BytesIO()

#         # The plot_skewt_from_json can accept a file path or file-like.
#         # We can save directly to the BytesIO, but we need a small tweak:
#         plot_skewt_from_json(parsed_data, output_filename=out_filename)

#         # Re-open that saved file from disk OR you can modify plot_skewt_from_json
#         # to accept a file-like object. For simplicity, let's assume it currently
#         # must save to disk, so let's read it back.
#         with open(out_filename, "rb") as f:
#             svg_data = f.read()

#         # Now upload the bytes to S3
#         s3_key = f"skew-outputs/{out_filename}"
#         try:
#             s3_client.put_object(
#                 Bucket=S3_BUCKET,
#                 Key=s3_key,
#                 Body=svg_data,
#                 ContentType="image/svg+xml"
#             )
#             # Construct a URL to the uploaded file (if the bucket is public or has a CloudFront distribution, etc.)
#             # For a non-public bucket, you’ll have to generate presigned URLs or set the ACL accordingly.
#             file_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
#             all_s3_urls.append(file_url)
#         except Exception as e:
#             print(f"Error uploading to S3: {e}")

#         # Clean up local file if needed
#         if os.path.exists(out_filename):
#             os.remove(out_filename)

#     # 5. Return some JSON describing where the files are stored
#     return jsonify({
#         "latitude": lat,
#         "longitude": lon,
#         "days_requested": days,
#         "s3_files": all_s3_urls,
#         "message": "SkewT images generated and uploaded to S3."
#     })

# # Minimal run call
# # If you’re running on an EC2 instance, you’d typically do something like
# # gunicorn --bind 0.0.0.0:5000 chart_server:app
# # or you can just do python chart_server.py for testing:
# if __name__ == "__main__":
#     # Make sure to allow inbound traffic on the port in your EC2 Security Group
#     app.run(host="0.0.0.0", port=5000, debug=True)
