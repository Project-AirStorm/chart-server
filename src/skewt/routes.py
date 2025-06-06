import datetime
import time
import requests
import boto3
from flask import Blueprint, request, jsonify
from .generate_skew import parse_json, plot_skewt

skewt_bp = Blueprint("skewt_bp", __name__)

# Initialize S3 Stuff
s3_client = boto3.client("s3", region_name="us-east-1")
BUCKET_NAME = "meteo-charts"  # S3 bucket
FOLDER_NAME = "skewt-svg-dumps"  # S3 bucket folder

@skewt_bp.route("/generate-skew", methods=["GET"])
def generate_skew():
    """
    Example endpoint: /generate-skew?days=3&lat=32.52&lon=-93.75
    """
    # Grab the query parameters
    forecast_days = request.args.get("days", type=int)
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    user_id = request.args.get("user_id", default="clerk-user")  # clerk user

    # Bomb out if a bad GET request was sent
    if forecast_days is None or lat is None or lon is None:
        return jsonify({"error": "Must provide 'days', 'lat', and 'lon'"}), 400

    # Build a subfolder name using chart_count + timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # e.g. "chart_1_2025-03-11_10-39-07"
    chart_folder = f"chart_{timestamp}"

    # 2. Build the Open-Meteo API URL based on these params
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&hourly="
        "temperature_1000hPa,temperature_975hPa,temperature_950hPa,temperature_925hPa,"
        "temperature_900hPa,temperature_875hPa,temperature_850hPa,temperature_825hPa,"
        "temperature_800hPa,temperature_775hPa,temperature_750hPa,temperature_725hPa,"
        "temperature_700hPa,temperature_675hPa,temperature_650hPa,temperature_625hPa,"
        "temperature_600hPa,temperature_575hPa,temperature_550hPa,temperature_525hPa,"
        "temperature_500hPa,temperature_475hPa,temperature_450hPa,temperature_425hPa,"
        "temperature_400hPa,temperature_375hPa,temperature_350hPa,temperature_325hPa,"
        "temperature_300hPa,temperature_275hPa,temperature_250hPa,temperature_225hPa,"
        "temperature_200hPa,temperature_175hPa,temperature_150hPa,temperature_125hPa,"
        "temperature_100hPa,temperature_70hPa,temperature_50hPa,temperature_40hPa,"
        "temperature_30hPa,temperature_20hPa,temperature_15hPa,temperature_10hPa,"
        "dew_point_1000hPa,dew_point_975hPa,dew_point_950hPa,dew_point_925hPa,"
        "dew_point_900hPa,dew_point_875hPa,dew_point_850hPa,dew_point_825hPa,"
        "dew_point_800hPa,dew_point_775hPa,dew_point_750hPa,dew_point_725hPa,"
        "dew_point_700hPa,dew_point_675hPa,dew_point_650hPa,dew_point_625hPa,"
        "dew_point_600hPa,dew_point_575hPa,dew_point_550hPa,dew_point_525hPa,"
        "dew_point_500hPa,dew_point_475hPa,dew_point_450hPa,dew_point_425hPa,"
        "dew_point_400hPa,dew_point_375hPa,dew_point_350hPa,dew_point_325hPa,"
        "dew_point_300hPa,dew_point_275hPa,dew_point_250hPa,dew_point_225hPa,"
        "dew_point_200hPa,dew_point_175hPa,dew_point_150hPa,dew_point_125hPa,"
        "dew_point_100hPa,dew_point_70hPa,dew_point_50hPa,dew_point_40hPa,"
        "dew_point_30hPa,dew_point_20hPa,dew_point_15hPa,dew_point_10hPa,"
        "wind_speed_1000hPa,wind_speed_975hPa,wind_speed_950hPa,wind_speed_925hPa,"
        "wind_speed_900hPa,wind_speed_875hPa,wind_speed_850hPa,wind_speed_825hPa,"
        "wind_speed_800hPa,wind_speed_775hPa,wind_speed_750hPa,wind_speed_725hPa,"
        "wind_speed_700hPa,wind_speed_675hPa,wind_speed_650hPa,wind_speed_625hPa,"
        "wind_speed_600hPa,wind_speed_575hPa,wind_speed_550hPa,wind_speed_525hPa,"
        "wind_speed_500hPa,wind_speed_475hPa,wind_speed_450hPa,wind_speed_425hPa,"
        "wind_speed_400hPa,wind_speed_375hPa,wind_speed_350hPa,wind_speed_325hPa,"
        "wind_speed_300hPa,wind_speed_275hPa,wind_speed_250hPa,wind_speed_225hPa,"
        "wind_speed_200hPa,wind_speed_175hPa,wind_speed_150hPa,wind_speed_125hPa,"
        "wind_speed_100hPa,wind_speed_70hPa,wind_speed_50hPa,wind_speed_40hPa,"
        "wind_speed_30hPa,wind_speed_20hPa,wind_speed_15hPa,wind_speed_10hPa,"
        "wind_direction_1000hPa,wind_direction_975hPa,wind_direction_950hPa,"
        "wind_direction_925hPa,wind_direction_900hPa,wind_direction_875hPa,"
        "wind_direction_850hPa,wind_direction_825hPa,wind_direction_800hPa,"
        "wind_direction_775hPa,wind_direction_750hPa,wind_direction_725hPa,"
        "wind_direction_700hPa,wind_direction_675hPa,wind_direction_650hPa,"
        "wind_direction_625hPa,wind_direction_600hPa,wind_direction_575hPa,"
        "wind_direction_550hPa,wind_direction_525hPa,wind_direction_500hPa,"
        "wind_direction_475hPa,wind_direction_450hPa,wind_direction_425hPa,"
        "wind_direction_400hPa,wind_direction_375hPa,wind_direction_350hPa,"
        "wind_direction_325hPa,wind_direction_300hPa,wind_direction_275hPa,"
        "wind_direction_250hPa,wind_direction_225hPa,wind_direction_200hPa,"
        "wind_direction_175hPa,wind_direction_150hPa,wind_direction_125hPa,"
        "wind_direction_100hPa,wind_direction_70hPa,wind_direction_50hPa,"
        "wind_direction_40hPa,wind_direction_30hPa,wind_direction_20hPa,"
        "wind_direction_15hPa,wind_direction_10hPa,"
        "geopotential_height_1000hPa,geopotential_height_975hPa,"
        "geopotential_height_950hPa,geopotential_height_925hPa,"
        "geopotential_height_900hPa,geopotential_height_875hPa,"
        "geopotential_height_850hPa,geopotential_height_825hPa,"
        "geopotential_height_800hPa,geopotential_height_775hPa,"
        "geopotential_height_750hPa,geopotential_height_725hPa,"
        "geopotential_height_700hPa,geopotential_height_675hPa,"
        "geopotential_height_650hPa,geopotential_height_625hPa,"
        "geopotential_height_600hPa,geopotential_height_575hPa,"
        "geopotential_height_550hPa,geopotential_height_525hPa,"
        "geopotential_height_500hPa,geopotential_height_475hPa,"
        "geopotential_height_450hPa,geopotential_height_425hPa,"
        "geopotential_height_400hPa,geopotential_height_375hPa,"
        "geopotential_height_350hPa,geopotential_height_325hPa,"
        "geopotential_height_300hPa,geopotential_height_275hPa,"
        "geopotential_height_250hPa,geopotential_height_225hPa,"
        "geopotential_height_200hPa,geopotential_height_175hPa,"
        "geopotential_height_150hPa,geopotential_height_125hPa,"
        "geopotential_height_100hPa,geopotential_height_70hPa,"
        "geopotential_height_50hPa,geopotential_height_40hPa,"
        "geopotential_height_30hPa,geopotential_height_20hPa,"
        "geopotential_height_15hPa,geopotential_height_10hPa"
        "&forecast_days=7"
        f"&forecast_days={forecast_days}"
        "&models=gfs_graphcast025"
    )
    # 3. Fetch the weather data from Open-Meteo
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        weather_json = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    hours_to_process = forecast_days * 24
    all_s3_urls = []

    # Iterate over hours and generate SVGs
    for hour in range(hours_to_process):
        start_time = time.time()

        parsed_data = parse_json(weather_json, hour_index=hour)

        # The plot_skewt_from_json can accept a file path or file-like.
        svg_data = plot_skewt(parsed_data, output_filename=None)

        # Now upload the bytes to S3

        s3_key = (
            f"skewt-diagrams/"
            f"skewt-dumps_{user_id}/"
            f"{chart_folder}/"
            f"skewt_hour_{hour:03d}_lat_{lat}_long_{lon}.svg"
        )

        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=svg_data,
                ContentType="image/svg+xml",
            )
            # Construct a URL to the uploaded file (if the bucket is public or has a CloudFront distribution, etc.)
            # For a non-public bucket, you’ll have to generate presigned URLs or set the ACL accordingly.
            print(f"Uploaded {s3_key} to S3 successfully.")
            file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
            all_s3_urls.append(file_url)

        except Exception as e:
            print(f"Error uploading to S3: {e}")

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Return JSON respone, describes where the files are stored
    return jsonify(
        {
            "latitude": lat,
            "longitude": lon,
            "user_id": user_id,
            "chart_folder": chart_folder,
            "days_requested": forecast_days,
            "s3_files": all_s3_urls,
            "message": "SkewT images generated and uploaded to S3.",
        }
    )
