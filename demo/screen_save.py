import cv2
import numpy as np
import pyautogui
import time
from pymavlink import mavutil
import constants



connection = mavutil.mavlink_connection('udp:localhost:14554')
connection.wait_heartbeat()

# Ekran çözünürlüğünü belirleyin (kendi ekran çözünürlüğünüzü kullanın)
screen_width, screen_height = pyautogui.size()
screen_size = (screen_width, screen_height)

# Video kaydı ayarları
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("scrren_output.mp4", fourcc, 30.0, screen_size)

# Kaydedilecek süre (saniye cinsinden)
duration = 500
end_time = time.time() + duration

while True:
    # Ekran görüntüsü al
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    current_altitude=constants.get_position_alt(connection)
    current_lat,current_lon=constants.get_position(connection)
    current_roll, current_pitch, current_yaw = constants.get_pitch_yaw_roll(connection)
    current_speed=constants.get_speed(connection)
    current_mode=constants.print_current_flight_mode(connection)
    # Metin ekleme
    text = f"Recording... {int(end_time - time.time())} seconds left"
    text+=f"Mode: {current_mode}"

    text+=f"Altitude: {current_altitude} m"
    text+=f"Pitch: {current_pitch}"
    text+=f"Speed: {current_speed} m/s"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Video dosyasına yaz
    out.write(frame)
    
    # Kayıt süresi doldu mu kontrol et
    if time.time() > end_time:
        break

# Her şeyi serbest bırak
out.release()
cv2.destroyAllWindows()
