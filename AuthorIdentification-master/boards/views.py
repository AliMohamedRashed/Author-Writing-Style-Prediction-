import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login
from django.http import HttpResponse
from .models import Board
# Create your views here.
def home(request):
    if  request.method == 'POST':
        articl = request.POST['text']
        print(articl)
        df = pd.read_csv("E:\\Train_test_v3.csv")
        df.head()
        X = df['article']
        y = df['author']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        class DenseTransformer(TransformerMixin):
            def fit(self, X, y=None, **fit_params):
                return self
            def transform(self, X, y=None, **fit_params):
                return X.todense()
        text_clf = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC()), ])

        text_clf.fit(X_train, y_train)
        predictions = text_clf.predict(X_test)
        print("Accuracy = {} %".format(metrics.accuracy_score(y_test, predictions) * 100))
        author =text_clf.predict([articl])[0]
        print(author)
        print(author)

        return render(request, 'result.html',{'author':author,'article':articl})
    return render(request,'home.html')
def about(request):
    return render(request,'about.html')
def result(request):
    if request.method == 'POST':
        return render(request, 'home.html')
def register(request):
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('index')
    return render(request,'register.html',{'form':form})