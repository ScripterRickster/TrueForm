from flask import Flask, request, render_template, redirect, url_for
import subprocess

app = Flask(__name__)

trackersRunning = {
    "pushups": False,
    "situps": False,
    "demo": False,
}

statTracker = {
    "pushups": 0,
    "situps": 0,
    "calories_burnt":0,
}

def trackPushups():
    if not trackersRunning["pushups"]:
        trackersRunning["pushups"] = True
        subprocess.run(['python3', 'Pushups.py'])
    return "yes"

def trackSitups():
    if not trackersRunning["situps"]:
        trackersRunning["situps"] = True
        subprocess.run(['python3', 'Situps.py'])
    return "yes"

def trackDemo():
    if not trackersRunning["demo"]:
        trackersRunning["demo"] = True
        subprocess.run(['python3', 'Tracker.py'])
    return "yes"

def updateStats(stat,val):
    if statTracker[stat]:
        statTracker[stat] = statTracker[stat]+val

def resetTracker(stat):
    if trackersRunning[stat]:
        for s in statTracker.keys():
            statTracker[s] = 0
        trackersRunning[stat] = False

def home():
    return render_template('index.html')
@app.route('/stats')
def stats():
    return render_template('stats.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        act = request.form.get('action')
        app.logger.info(f"Received POST request with action: {act}")  # Added logging

        if act == 's_pushup':
            trackPushups()
            print("pushup")
        elif act == 's_situp':
            trackSitups()
            print("situp")
        elif act == 's_demo':
            trackDemo()
            print("demo")

        return redirect(url_for('index'))

    return render_template('index.html')



#if __name__ == '__main__':
    #app.run(debug=True)
