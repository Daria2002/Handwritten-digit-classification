FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install numpy
RUN pip install matplotlib
RUN pip install tensorflow
RUN pip install tensorflow
ENTRYPOINT ["python"]
CMD ["src/app.py"]
