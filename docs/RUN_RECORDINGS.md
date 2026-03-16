# Running analysis on recorded videos & viewing anomaly reports

You have recordings under:
- `E:\vision-ai\vision-ai\data\recordings\Office_Corridor`
- `E:\vision-ai\vision-ai\data\recordings\Storage_Area`

And 4 employees registered. Follow these steps to run the pipeline on those videos and see anomaly reports.

---

## 1. Get camera IDs (Office Corridor & Storage Area)

The app needs a **camera_id** for each recording folder. Those cameras are in the database (from seed or API).

- Open: **http://localhost:8000/api/cameras/**
- Find the entries whose **name** is "Office Corridor" and "Storage Area" and note their **id** (e.g. `4` and `3`).

If you don’t have those cameras, add them via **Dashboard** or:

```http
POST /api/cameras/
Content-Type: application/json

{"name": "Office Corridor", "location": "Admin Building", "rtsp_url": "file:///E:/vision-ai/vision-ai/data/recordings/Office_Corridor/rec_2026-01-01_00-00-00.mp4"}
```

(You can change `rtsp_url` to any default; when running from a file we pass `source_url`.)

---

## 2. Pick one video file per folder

The pipeline runs on **one video file** at a time (e.g. one `.mp4` per camera).

- In PowerShell:
  ```powershell
  dir "E:\vision-ai\vision-ai\data\recordings\Office_Corridor\*.mp4"
  dir "E:\vision-ai\vision-ai\data\recordings\Storage_Area\*.mp4"
  ```
- Note one filename from each folder (e.g. `rec_2026-03-05_06-00-00.mp4`).

Use **forward slashes** in URLs. Example full paths:

- `E:/vision-ai/vision-ai/data/recordings/Office_Corridor/rec_2026-03-05_06-00-00.mp4`
- `E:/vision-ai/vision-ai/data/recordings/Storage_Area/rec_2026-03-05_06-00-00.mp4`

---

## 3. Start the pipeline on each recording

Start the pipeline **with a recording file** (not live RTSP) by passing `source_url`:

Replace `OFFICE_CAMERA_ID` and `STORAGE_CAMERA_ID` with the ids from step 1.  
Replace the file names with your actual `.mp4` names.

**Option A – Browser / API docs**

- Open **http://localhost:8000/docs**
- **POST /api/pipeline/start/{camera_id}**
- Set `camera_id` to the Office Corridor camera id.
- Add query parameter: `source_url` =  
  `E:/vision-ai/vision-ai/data/recordings/Office_Corridor/rec_YYYY-MM-DD_HH-MM-SS.mp4`
- Execute. Then repeat for Storage Area with its `camera_id` and the Storage_Area file path.

**Option B – PowerShell (curl)**

```powershell
# Office Corridor (use your camera_id and filename)
curl -X POST "http://localhost:8000/api/pipeline/start/OFFICE_CAMERA_ID?source_url=E:/vision-ai/vision-ai/data/recordings/Office_Corridor/rec_2026-03-05_06-00-00.mp4"

# Storage Area
curl -X POST "http://localhost:8000/api/pipeline/start/STORAGE_CAMERA_ID?source_url=E:/vision-ai/vision-ai/data/recordings/Storage_Area/rec_2026-03-05_06-00-00.mp4"
```

**Option C – Dashboard**

- Go to **Dashboard** → Camera Feeds.
- For each camera card you can’t pass a file path from the UI. So for **recordings**, use the API (A or B) above. The Dashboard “Start” uses the camera’s stored `rtsp_url` (live or a default file).

The pipeline will:

- Read frames from the video file
- Run person detection (YOLO) and optional employee recognition (your 4 employees)
- Track people and check zones / idle / staffing
- Write **anomaly events** to the database
- Stop when the file ends (no new recording is created when the source is a file)

---

## 4. Where to see anomaly reports and full details

### A. Dashboard – Recent Events (live list)

- **http://localhost:8000/** → section **“Recent Events”**
- Table: Time, Camera, Type, Severity, Description, Ack.
- Use filters: **All Types** / **All Severities** (or narrow by type/severity).
- This shows **all** events; after you run the recordings, new rows appear here.

### B. Reports page – saved anomaly report (period + full details)

- **http://localhost:8000/reports**
- **Generate & save report**
  - Set **Period start** and **Period end** to cover the time when you ran the recordings (e.g. today 00:00 to now).
  - Optional **Report title** (e.g. “Office & Storage 05-Mar”).
  - Click **Save report**.
- **Saved reports**
  - List of saved reports (title, period, number of anomalies).
  - Click **View** on a report to see:
    - Summary (period, total anomalies, cameras, count by type)
    - **Full event table**: Time, Camera, Type, Severity, Description for every event in that period.

### C. API – events and reports

- **List events (filterable):**  
  `GET /api/events/?camera_id=4&limit=100`
- **Event count:**  
  `GET /api/events/count?start_time=...&end_time=...`
- **List saved reports:**  
  `GET /api/reports/`
- **One report with full event list:**  
  `GET /api/reports/{report_id}`

---

## 5. Quick checklist

1. Get **camera_id** for “Office Corridor” and “Storage Area” from **GET /api/cameras/**.
2. Choose **one .mp4** per folder under `data/recordings/Office_Corridor` and `Storage_Area`.
3. **Start pipeline** for each camera with `source_url` = full path to that .mp4 (e.g. `POST /api/pipeline/start/4?source_url=E:/vision-ai/vision-ai/data/recordings/Office_Corridor/rec_xxx.mp4`).
4. Watch **Dashboard → Recent Events** for new anomalies (filter by type e.g. **Unknown Person Sighting** to see who/when).
5. Open **Reports** → set period → **Save report** → **View** for a full anomaly report and event details.

**Shift-based recognition:** Only employees on the **current shift** (from Shift Schedules) are treated as known; anyone else is reported as **unknown person**. Assign each employee to a shift in **Employees** so they are recognised when that shift is active. **Unknown person sighting** events record the exact time and track ID for each unknown person.
