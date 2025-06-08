import socket

# ตั้งค่า IP และ PORT ของ Raspberry Pi
HOST = "192.168.1.101"  # เปลี่ยนเป็น IP ของ Pi
PORT = 12345

message = input("Enter text : ")

# เชื่อมต่อกับ Raspberry Pi และส่งข้อความ
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(message.encode())  # ส่งข้อความ
    print("✅ Done!")
