build:
	cd docker && docker compose up -d --build

server:
	uvicorn api:app --reload --port 5555

# change API_URL to http://127.0.0.1:5555/crop-image/ in web.py before before run the script
client:
	python web.py 
	