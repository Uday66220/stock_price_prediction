from flask import Flask,render_template,request
import numpy as np
import pickle
asian=pickle.load(open("models/Asian.pkl","rb"))
bpcl=pickle.load(open("models/BPCL.pkl","rb"))
cipla=pickle.load(open("models/Cipla.pkl","rb"))
itc=pickle.load(open("models/ITC.pkl","rb"))
it=pickle.load(open("models/LT.pkl","rb"))
sbin=pickle.load(open("models/SBI.pkl","rb"))
tatamotors=pickle.load(open("models/Tatamotors.pkl","rb"))
tcs=pickle.load(open("models/TCS.pkl","rb"))
titan=pickle.load(open("models/Titan.pkl","rb"))
wipro=pickle.load(open("models/Wipro.pkl","rb"))
stocks={"Asian":asian,"BPCL":bpcl,"Cipla":cipla,"ITC":itc,"IT":it,"SBI":sbin,"TataMotors":tatamotors,"TCS":tcs,"Titan":titan,"Wipro":wipro}
app=Flask(__name__)
@app.route("/",methods=["POST","GET"])
def home():
    return render_template("index.html")
@app.route("/submit",methods=["POST"])
def result():
    l=[]
    company=request.form["company"]
    prevclose=float(request.form["prevclose"])
    open=float(request.form["open"])
    high=float(request.form["high"])
    low=float(request.form["low"])
    last=float(request.form["last"])
    l=[prevclose,open,low,high,last]
    l=np.array(l,dtype="float64").reshape((1,-1))
    pred=stocks[company].predict(l)
    return render_template("predict.html",comp=company,prevclose=prevclose,open=open,high=high,low=low,last=last,data=float(pred))
if __name__=="__main__":
    app.run(debug=True)