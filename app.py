# Importing various packages for Flask App. Also importing sklearn library for implementing tfidf. Note - Flask is a Python based framework for developing web Applications.

from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from yake import KeywordExtractor

# Initialize Flask App
app = Flask(__name__)
# Add configuration for type of db used
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendees.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Initialize SQLAlchemy - It is an object Relational Mapper for SQLite
db = SQLAlchemy(app)

# Define Database Models : 
class Documents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doc_text = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return '<Documents %r>' % self.id

# This is the model for the Events Table
class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_name = db.Column(db.String(200), nullable=False)
    event_host = db.Column(db.String(200), nullable=False)
    event_agenda = db.Column(db.String(200), nullable=False)
    attendees = db.relationship('Attendee', backref='owner')
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Event %r>' % self.id


# Model for the Attendee Table
class Attendee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    custom_id = db.Column(db.Integer(), nullable=False)
    attendee_name = db.Column(db.String(200), nullable=False)
    attendee_email = db.Column(db.String(200), nullable=False)
    attendee_company = db.Column(db.String(200), nullable=False)
    attendee_description = db.Column(db.String(200), nullable=False)
    attendee_sector = db.Column(db.String(200), nullable=False)
    match_description_predict = db.Column(db.String(200), nullable=True, default = " ")
    match_description_email = db.Column(db.String(200), nullable=True,  default = " ")
    match_role_predict = db.Column(db.String(200), nullable=True, default = " ")
    match_role_email = db.Column(db.String(200), nullable=True,  default = " ")
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    attendee_role = db.Column(db.String(200), nullable=False)
    attendee_experience = db.Column(db.Integer(), nullable = False)
    owner_id = db.Column(db.Integer, db.ForeignKey('event.id'))
    keywords = db.Column(db.String(200), nullable=True, default = " ")
    add_keywords = db.Column(db.String(200), nullable=True, default = " ")
    

    def __repr__(self):
        return '<Attendee %r>' % self.id

# Base Route for the application for rendering home page
@app.route('/')
def home():
    events = Event.query.order_by(Event.date_created).all()
    return render_template('index.html', events=events)

# Route for Event Dashboard
@app.route('/<int:event_id>/dashboard')
def index(event_id):
    curr_event = Event.query.get_or_404(event_id)
    all_events = Event.query.order_by(Event.date_created).all()
    return render_template('dashboard.html', event = curr_event, all_events = all_events )

# Route for create Event
@app.route('/createevent', methods=['GET','POST'])
def createevent():
    if request.method == 'POST':
        # get event details as entered by user from html form
        event_name = request.form['event-name']
        event_host = request.form['event-host']
        event_agenda = request.form['event-agenda']
        new_event = Event(event_name = event_name, event_host = event_host, event_agenda = event_agenda)

        try:
            db.session.add(new_event)
            db.session.commit()
            return redirect('/')
        except:
            return "There was an issue adding the task"

    else:
        return render_template('createevent.html')

# Route for delete event
@app.route('/<int:event_id>/delete')
def delete_event(event_id):
    # get event to delete from db
    event_to_delete = Event.query.get_or_404(event_id)
    deleted_events = Attendee.query.filter_by(owner_id = event_id).delete()
    try:
        db.session.delete(event_to_delete)
        # commit changes to db
        db.session.commit()
        # redirect to homepage
        return redirect('/')
    except:
        return 'There was a problem deleting that task'

# Event Details page
@app.route('/<int:event_id>/eventdetails')
def eventdetails(event_id):
    curr_event = Event.query.get_or_404(event_id)
    return render_template('eventdetails.html', event = curr_event )

# Attendees for event given by event id
@app.route('/<int:event_id>/attendees')
def attendees(event_id):
    curr_event = Event.query.get_or_404(event_id)
    attendees = Attendee.query.order_by(Attendee.date_created).filter_by(owner_id=event_id)
    # render the attendees template
    return render_template('attendees.html', event = curr_event, attendees=attendees)

# delete a particular attendee identified by id
@app.route('/<int:event_id>/attendees/delete/<int:attendee_id>')
def delete(event_id, attendee_id):
    attendee_to_delete = Attendee.query.get_or_404(attendee_id)

    try:
        db.session.delete(attendee_to_delete)
        db.session.commit()
        return redirect('/{}/attendees'.format(event_id))
    except:
        return 'There was a problem deleting that task'

# update a particular attendee identified by id
@app.route('/<int:event_id>/attendees/update/<int:attendee_id>', methods=['GET', 'POST'])
def update_attendee(event_id, attendee_id):
    attendee_to_update = Attendee.query.get_or_404(attendee_id)
    curr_event = Event.query.get_or_404(event_id)
    if request.method == 'POST':
        attendee_to_update.attendee_name = request.form['attendee-name']
        attendee_to_update.attendee_email = request.form['attendee-email']
        attendee_to_update.attendee_company = request.form['attendee-company']
        attendee_to_update.attendee_description = request.form['attendee-description']
        attendee_to_update.attendee_experience = request.form['attendee-experience']
        attendee_to_update.attendee_role = request.form['attendee-role']
        attendee_to_update.attendee_role = request.form['add-keywords']

        try:
            db.session.commit()
            return redirect('/{}/attendees'.format(event_id))
        except:
            return 'There was an issue updating your task'

    else:
        return render_template('updateattendee.html', curr_event = curr_event, attendee = attendee_to_update)

# Register new attendee
@app.route('/<int:event_id>/registerattendee', methods=['GET','POST'])
def add_attendee(event_id):
    curr_event = Event.query.get_or_404(event_id)
    acount = Attendee.query.filter_by(owner_id=event_id).count()
    if request.method == 'POST':
        custom_id = acount+1
        attendee_name = request.form['attendee-name']
        attendee_email = request.form['attendee-email']
        attendee_company = request.form['attendee-company']
        attendee_description = request.form['attendee-description']
        attendee_experience = request.form['attendee-experience']
        attendee_role = request.form['attendee-role']
        attendee_sector = request.form['attendee-sector']

        # Following lines are for keyword extraction  :
        # first, generate text to be extracted keywords from.
        text = attendee_description
        # create an object of the KeywordExtractor class
        kw_extractor = KeywordExtractor(lan="en", n=1, top=10)
        # store generated keywords
        keywords = kw_extractor.extract_keywords(text=text)
        keywords = [x for x, y in keywords]
        # the keywords are in a list, so we have to convert them to string.
        str = " "
        # using python join for converting to string
        attendee_keywords = str.join(keywords)
        # final keywords to be added to db
        attendee_keywords = attendee_keywords +" "+ attendee_role +" "+ attendee_sector
        print(attendee_keywords)
        new_attendee = Attendee(custom_id = custom_id, attendee_name = attendee_name, attendee_email = attendee_email, attendee_company = attendee_company, attendee_description = attendee_description, attendee_experience = attendee_experience, attendee_role = attendee_role, attendee_sector = attendee_sector, keywords = attendee_keywords,  owner = curr_event)

        try:            
            db.session.add(new_attendee)
            db.session.commit()
            # under process ----- the code is likely to be changed.
            return redirect('/{}/registerfinal/{}'.format(event_id,new_attendee.id))
        except:
            return "There was an issue adding the task in registerattendee"

    else:
        return render_template('registerattendee.html', curr_event = curr_event)

# modular code ---------- likely to be changed
@app.route('/<int:event_id>/registerfinal/<int:attendee_id>', methods = ['GET','POST'])
def add_final_attendee(event_id, attendee_id):
    attendee = Attendee.query.get_or_404(attendee_id)
    curr_event = Event.query.get_or_404(event_id)
    if request.method == 'POST':
        attendee.add_keywords = request.form['add-keyword'] 
        try:
            db.session.commit()
            return redirect('/{}/dashboard'.format(event_id))
        except:
            return 'There was an issue updating your task'
    else:
        return render_template('registerfinal.html',curr_event = curr_event, attendee = attendee)


# matchmaking route, list all the potential matches.
@app.route('/<int:event_id>/matchmaking')
def matchmaking(event_id):
    curr_event = Event.query.get_or_404(event_id)
    attendees = Attendee.query.order_by(Attendee.date_created).filter_by(owner_id=event_id)
    return render_template('matchmaking.html', event = curr_event, attendees=attendees)

# Predict route for the matchmaking
@app.route('/<int:event_id>/matchmaking/predict/<int:attendee_id>')
def predict(event_id, attendee_id):

    attendee = Attendee.query.get_or_404(attendee_id)
    curr_event = Event.query.get_or_404(event_id)
    query = db.session.query(Attendee).filter( attendee.owner_id == event_id).all()
    print(query)
    # generating tfidf documents
    documents_check = []
    for every in query:
        documents_check.append(every.keywords+" "+every.attendee_sector+" "+every.attendee_company +" "+ every.add_keywords)
        print(every.attendee_name)
    print(documents_check)

  #  individual_text_for_role = attendee.attendee_role + " " + attendee.attendee_company
    individual_text = attendee.keywords + " " + attendee.attendee_sector + " " + attendee.attendee_company + " " + attendee.add_keywords
  #  print("the individual text is" + individual_text_for_role)
    print("the attendee id is {}".format(attendee_id))
    id_to_send = attendee_id-1
    match_id = getMatch(id_to_send, individual_text, documents_check) + 1
    print("Now printing match id! = ")
    
    match = Attendee.query.get_or_404(match_id)

    attendee.match_description_predict = match.attendee_name
    attendee.match_description_email = match.attendee_email

    try:
        print('commiting to db')
        db.session.commit()
        return redirect('/{}/matchmaking'.format(event_id))
    except:
        return 'There was an issue updating your task'


# Generate similarity array, for each document, then return the second document, for the closest 
def most_similar(doc_id,similarity_matrix,matrix):
    similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    print(similar_ix)
    return similar_ix[1]
    
# matchmaking function
def getMatch(index, text, documents):
    print("the documents are {}".format(documents))
    pd.set_option('display.max_colwidth', 0)
    pd.set_option('display.max_columns', 0)
    documents_df=pd.DataFrame(documents,columns=['documents'])
    print(documents_df)

    # removing special characters and stop words from the text
    stop_words_l=stopwords.words('english')
    # cleaning dataset
    documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
    # vectorization process -- 
    tfidfvectoriser=TfidfVectorizer(max_features=64)
    tfidfvectoriser.fit(documents_df.documents_cleaned)
    tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)
    tfidf_vectors=tfidf_vectors.toarray()

    # Compute pairwise similarity
    pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
    prediction = most_similar(index,pairwise_similarities,'Cosine Similarity')
    print(prediction)
    # return prediction to main function
    return prediction.item()



if __name__ == "__main__":
    app.run(debug=True)

    