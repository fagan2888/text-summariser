from app import app
from waitress import serve

if __name__ == "__main__":
    app.debug = False
    serve(app, host='0.0.0.0', port=8000)
   # app.run(host='0.0.0.0', port=80)