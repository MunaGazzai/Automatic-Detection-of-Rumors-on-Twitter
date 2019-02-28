# tag=1 (tweet is real) tag=0 (tweet is fake)
x=Tweet_with_Tag['text']
y=Tweet_with_Tag['tag']

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=.20, random_state=0)

## bag of words ##
cv= CountVectorizer()
countx=cv.fit_transform(X_train)
#countx.shape
counttest=cv.transform(X_test)
#counttest.shape

## BOW & TF-IDF ##
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(countx)
# X_train_tfidf.shape
counttest=cv.transform(X_test)
#counttest.shape
X_test_tfidf=tfidf_transformer.transform(counttest)
#X_test_tfidf.shape



## KNeighborsClassifier With BOW ##
# search for an optimal value of K for KNN
K_rang=range(1,50)
K_score=[]
for k in K_rang:
    KNN=KNeighborsClassifier(n_neighbors=k)
    KNN.fit(countx, Y_train)
    predicted = KNN.predict(counttest)
    K_score.append(accuracy_score(Y_test, predicted))
print(K_score)

#polt the value of K for KNN
plt.plot(K_rang,K_score)
plt.xlabel("value of k for knn")
plt.ylabel("cross-valdition Accuracy")
# 50 is best

#build the model
KNN=KNeighborsClassifier(n_neighbors=50)
KNN.fit(countx, Y_train)
predicted = KNN.predict(counttest)
print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))  
print(accuracy_score(Y_test, predicted))
#accuracy_score is 0.7428571428571429


# KNeighborsClassifier with BOW and TF-IDF
# search for an optimal value of K for KNN
K_rang=range(1,50)
K_score=[]
for k in K_rang:
    KNN=KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train_tfidf, Y_train)
    predicted = KNN.predict(X_test_tfidf)
    K_score.append(accuracy_score(Y_test, predicted))
print(K_score)

#polt the value of K for KNN
plt.plot(K_rang,K_score)
plt.xlabel("value of k for knn")
plt.ylabel("cross-valdition Accuracy")
# 37 is the best

#build the model
KNN=KNeighborsClassifier(n_neighbors=37)
KNN.fit(X_train_tfidf, Y_train)
predicted = KNN.predict(X_test_tfidf)


print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))  
print(accuracy_score(Y_test, predicted))
#accuracy_score is 0.8

##############################################

##MultinomialNB With BOW ##

NB=MultinomialNB()
NB.fit(countx, Y_train)
predicted = NB.predict(counttest)

print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))  
print(accuracy_score(Y_test, predicted))
# accuracy_score is 0.84

## MultinomialNB with BOW and TF-IDF ##

NB=MultinomialNB()
NB.fit(X_train_tfidf, Y_train)
predicted = NB.predict(X_test_tfidf)

print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))  
print(accuracy_score(Y_test, predicted))
# accuracy_score is 0.7542857142857143

##############################################

# #  Random ForestClassifier With BOW ##

RF=RandomForestClassifier(n_estimators=1000, random_state=0)
RF.fit(countx, Y_train)
predicted = RF.predict(counttest)

print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))  
print(accuracy_score(Y_test, predicted))
# accuracy_score is 0.8228571428571428

# #  Random ForestClassifier BOW and TF-IDF ##
RF=RandomForestClassifier(n_estimators=1000, random_state=0) 
RF.fit(X_train_tfidf, Y_train)
predicted = RF.predict(X_test_tfidf)

print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))  
print(accuracy_score(Y_test, predicted))

# accuracy_score is 0.8057142857142857

##############################################
# # support vector machine With BOW ##
svm = SVC(kernel='linear') 
svm.fit(countx, Y_train)
predicted = svm.predict(counttest)

print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))
print(accuracy_score(Y_test, predicted))

# accuracy_score is 0.8228571428571428

# #  support vector machine BOW and TF-IDF ##

svm = SVC(kernel='linear') 
svm.fit(X_train_tfidf, Y_train)
predicted = svm.predict(X_test_tfidf)

print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))
print(accuracy_score(Y_test, predicted))

# accuracy_score is 0.8342857142857143

##############################################

## GradientBoostingClassifier with BOW

GB = GradientBoostingClassifier(n_estimators=1000)
GB.fit(countx, Y_train)
predicted = GB.predict(counttest)

print(confusion_matrix(Y_test, predicted))  
print(classification_report(Y_test, predicted))  
print(accuracy_score(Y_test, predicted))
# accuracy_score is 0.8114285714285714
#Accuracy changes when I rebuild the model 


## GradientBoostingClassifierwith BOW and TF-IDF
GB = GradientBoostingClassifier(n_estimators=654)
GB.fit(X_train_tfidf, Y_train)
predicted = GB.predict(X_test_tfidf)
# accuracy_score is .8





