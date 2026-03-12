# License Plate Detection

## Local Run
python detect_plate.py [image.jpg]

## Deploy to Render
1. git init
2. git add .
3. git commit -m \"initial\"
4. git remote add origin https://github.com/YOURUSER/repo.git (create repo)
5. git push -u origin main
6. Render.com → New Web Service → Connect GitHub repo
7. Build: Python, Start: python gradio_ui.py

Live at yourapp.onrender.com - upload vehicle image, detect plates!
