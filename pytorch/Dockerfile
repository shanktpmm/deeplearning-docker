FROM python:3.10
RUN apt update && apt install -y python3-pip
RUN pip3 install numpy torch
COPY main.py app/
CMD ["python3", "app/main.py" ]
