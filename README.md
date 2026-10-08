# LicensePDet — Thai License Plate Detection

ระบบอ่านป้ายทะเบียนรถไทยจากกล้อง CCTV (RTSP / ไฟล์วิดีโอ / เว็บแคม)
Detects cars, tracks them, and reads the plate when a car crosses a line. Results go to SQLite and (optionally) an LCD.

```
กล้อง → YOLO11 หารถ → DeepSORT ให้ ID → รถข้ามเส้น → crop → หาป้าย → อ่านตัวอักษร
      → บันทึก SQLite (data/plates.db) + รูป (data/crops/) + ส่งขึ้นจอ LCD (ถ้าตั้งไว้)
```

## Quick start (clone → run)

ต้องมี Python 3.10+ (แนะนำ 3.11)

```bash
git clone https://github.com/Iliketoeatsalmon/LicensePlate_detect_system.git
cd LicensePlate_detect_system

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # ลง PyTorch ด้วย ใช้เวลาสักครู่

cp .env.example .env                                  # แล้วแก้ RTSP_URL เป็นกล้องของคุณ
lpr check                                             # เช็กโมเดล + ต่อกล้องได้ไหม
lpr run                                               # เริ่มทำงาน (กด q ในหน้าต่างเพื่อออก)
```

`.env` ขั้นต่ำ:

```
RTSP_URL=rtsp://USER:PASSWORD@192.168.1.50:554/video
```

> URL ของกล้องแต่ละยี่ห้อต่างกัน (เช่น `/stream1`, `/Streaming/Channels/101`) ดูในคู่มือกล้อง
> รหัสผ่านถูกซ่อนเป็น `***` ใน log ทุกที่ และ `.env` ไม่ถูก commit

## ตั้งเส้นนับรถ

รถจะถูกอ่านป้าย **ครั้งเดียว** ตอนจุดกึ่งกลางรถข้ามเส้น ตั้งใน `.env` เป็นสัดส่วนของภาพ (0–1):

```
LINE=0.1,0.75,0.9,0.75     # x1,y1,x2,y2  ค่าเริ่มต้น = เส้นแนวนอนที่ 75% ของความสูง
```

เปิดหน้าต่างพรีวิว (`lpr run` ไม่ใส่ `--no-show`) จะเห็นเส้นสีแดง ปรับจนอยู่จุดที่เห็นป้ายชัดที่สุด
ถ้ารถมาจากด้านบนของภาพ ให้เลื่อนเส้นลงมา ถ้ามาจากด้านล่างให้เลื่อนขึ้น

## คำสั่ง

| คำสั่ง | ทำอะไร |
|--------|--------|
| `lpr run [--source X] [--no-show]` | อ่านป้ายแบบต่อเนื่อง `X` = rtsp URL, ไฟล์วิดีโอ หรือเลขกล้อง (`0`) |
| `lpr check` | ตรวจโมเดล, device (cuda/mps/cpu), การต่อกล้อง |
| `lpr read-images [path…]` | อ่านป้ายจากรูป (ค่าเริ่มต้น: โฟลเดอร์ `samples/`) |
| `lpr history [-n 20]` | ดูป้ายล่าสุดที่บันทึกไว้ |

ตัวเลือกใน `.env`: `DEVICE` (auto/cuda/mps/cpu), `LCD_HOST`/`LCD_PORT` (ว่างไว้ = ปิดจอ LCD), `DB_PATH`, `CROP_DIR`, `MODELS_DIR` ดู `.env.example`

### จอ LCD (Raspberry Pi)

ตั้ง `LCD_HOST=<IP ของ Pi>` ระบบจะส่งข้อความ `ทะเบียน จังหวัด` (UTF-8) ผ่าน TCP ไปยัง `LCD_PORT` ทุกครั้งที่อ่านป้ายได้
ถ้าต่อ Pi ไม่ได้ ระบบแค่เตือนแล้วทำงานต่อ

## โครงสร้าง

```
src/lpr/      config · pipeline · track · plate · thai · store · display · overlay · cli
models/       yolo11n.pt (รถ) · license_plate.pt (ป้าย) · data_plate.pt (ตัวอักษร)
assets/       ฟอนต์ไทย
samples/      รูปป้ายตัวอย่าง
tests/        pytest (ไม่ต้องใช้กล้อง/GPU)
docs/         spec และแผนงาน
```

## พัฒนา / ทดสอบ

```bash
pip install -e ".[dev]"
pytest
lpr read-images        # smoke test ด้วยรูปตัวอย่าง
```

## ข้อจำกัดที่รู้อยู่

- ตาราง `src/lpr/thai.py` มีรหัสจังหวัดที่น่าสงสัย (`BTG`, `MSN`, `NPT` ซ้ำกับ `NBP`, `CNT` ซ้ำกับ `CRI`) ยังไม่แก้ เพราะโมเดลอาจใช้รหัสเหล่านี้จริง
- ต้องปรับ `LINE` ให้เหมาะกับมุมกล้องแต่ละที่
- ทดสอบกับ RTSP จำลอง (mediamtx) และวิดีโอสังเคราะห์ ยังไม่ได้ทดสอบกับกล้องจริงหลายยี่ห้อ
