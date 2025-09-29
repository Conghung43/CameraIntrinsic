import time
import requests
import datetime

URL = "https://104.43.119.169:5001/v1/sol_server/orders?status=PENDING&date=2025-08-29&number=20&trans_type_id=2,3,4,5"

# Disable SSL warnings (if server uses self-signed certs)
requests.packages.urllib3.disable_warnings()

previous_data = None

while True:
    time.sleep(5)  # wait 5 seconds before each request

    try:
        response = requests.get(URL, verify=False, timeout=10)
        response.raise_for_status()  # raise exception if bad response

        result = response.json()
        current_data = result.get("data", [])

        if current_data != previous_data:
            log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{log_time}] Data changed:")
            print(current_data)
            previous_data = current_data

    except Exception as e:
        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{log_time}] Error: {e}")