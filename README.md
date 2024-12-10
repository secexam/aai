### Exp 1 - BAYESIAN NETWORKS
```py
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.dag import Edge,EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
pd.options.display.max_columns=50

df=pd.read_csv('weatherAUS.csv',encoding='utf-8')

df=df[pd.isnull(df['RainTomorrow'])==False]

df=df.fillna(df.mean())

df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40'   if x<=40 else '1.40-50' if 40<x<=50 else '2.>50')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
print(df)

def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Caclucate probabilities
                prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                # Caclucate probabilities
                prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob

H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)
pos={0: (-1,0), 1: (-1, 0.5), 2: (1, 0), 3:(0,-0.5)}

options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "yellow",
    "edgecolors": "green",
    "edge_color": "black",
    "linewidths": 5,
    "width": 5,}


n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()
```

### Exp 2 - INFERENCE METHOD OF BAYESIAN NETWORK
```py
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

alarm_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.999], [0.001]]
)
cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.998], [0.002]]
)
cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
)
cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],
)
cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.99, 0.3], [0.01, 0.7]],
    evidence=["Alarm"],
    evidence_card=[2],
)

alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls
)

alarm_model.check_model()
inference=VariableElimination(alarm_model)
evidence={"JohnCalls":1,"MaryCalls":0}
query='Burglary'
res=inference.query(variables=[query],evidence=evidence)
print(res)
evidence2={"JohnCalls":1,"MaryCalls":1}
res2=inference.query(variables=[query],evidence=evidence2)
print(res2)
```

### Exp - 3 - APPROXIMATE INFERENCE IN BAYESIAN NETWORKS
```py
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import GibbsSampling
import networkx as nx
import matplotlib.pyplot as plt

network=BayesianNetwork([
    ('Burglary','Alarm'),
    ('Earthquake','Alarm'),
    ('Alarm','JohnCalls'),
    ('Alarm','MaryCalls')
])

cpd_burglary = TabularCPD(variable='Burglary',variable_card=2,values=[[0.999],[0.001]])
cpd_earthquake = TabularCPD(variable='Earthquake',variable_card=2,values=[[0.998],[0.002]])
cpd_alarm = TabularCPD(variable ='Alarm',variable_card=2, values=[[0.999, 0.71, 0.06, 0.05],[0.001, 0.29, 0.94, 0.95]],evidence=['Burglary','Earthquake'],evidence_card=[2,2])
cpd_john_calls = TabularCPD(variable='JohnCalls',variable_card=2,values=[[0.95,0.1],[0.05,0.9]],evidence=['Alarm'],evidence_card=[2])
cpd_mary_calls = TabularCPD(variable='MaryCalls',variable_card=2,values=[[0.99,0.3],[0.01,0.7]],evidence=['Alarm'],evidence_card=[2])

network.add_cpds(cpd_burglary,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_mary_calls)

print("Bayesian Network Structure :")
print(network)

G = nx.DiGraph()

nodes=['Burglary','Earthquake','Alarm','JohnCalls','MaryCalls']
edges=[('Burglary','Alarm'),('Earthquake','Alarm'),('Alarm','JohnCalls'),('Alarm','MaryCalls')]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

pos ={
    'Burglary':(0,0),
    'Earthquake':(2,0),
    'Alarm':(1,-2),
    'JohnCalls':(0,-4),
    'MaryCalls':(2,-4)
}

nx.draw(G,pos,with_labels=True,node_size=5000,node_color='yellow',font_size=10,font_weight='bold',arrowsize=12)
plt.title("Bayesian Network: Alarm Problem")
plt.show()

gibbs_sampler =GibbsSampling(network)
num_samples=10000

samples=gibbs_sampler.sample(size=num_samples)

query_variable='Burglary'
query_result= samples[query_variable].value_counts(normalize=True)

print("\n Approximate Probabilities of {}".format(query_variable))
print(query_result)
```

### Exp - 4 - Implementation of Hidden Markov Model
```py
import numpy as np
t_m = np.array([[0.7,0.3],[0.4, 0.6]])
e_m = np.array([[0.1,0.9],[0.8,0.2]])
i_p = np.array([0.5,0.5])
o_s = np.array([1,1,1,0,0,1])

alpha = np.zeros((len(o_s),len(i_p)))
alpha[0, :] = i_p * e_m[:, o_s[0]]

for t in range(1, len(o_s)):
  for j in range(len(i_p)):
    alpha[t, j ] = e_m[j,o_s[t]] * np.sum(alpha[t-1, :] * t_m[:, j ])

prob = np.sum(alpha[-1, :])
print("The Probability of the observed sequence is :",prob)

m_l_s = []
for t in range(len(o_s)):
  if alpha[t, 0] > alpha[t,1]:
    m_l_s.append("sunny")
  else:
    m_l_s.append("rainy")
print("Thye most likely sequence of weather states is :",m_l_s)
```

### Exp - 5 - Kalman Filter
```py
import numpy as np
import matplotlib.pyplot as plt
class KalmanFi1ter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F=F
        self.H=H
        self.Q=Q
        self.R=R
        self.x=x0
        self.P=P0
    def predict (self):
        self.x=np.dot(self.F,self.x)
        self.P=np.dot(np.dot(self. F,self. P),self.F.T)+self.Q
    def update(self,z):
        y=z-np.dot(self.H,self.x)
        s=np.dot(np.dot(self.H,self.P),self.H.T)+self.R
        K=np.dot(np.dot(self.P,self.H.T),np.linalg.inv(s))
        self.x=self.x+np.dot(K,y)
        self.P=np.dot(np.eye(self.F.shape[0])-np.dot(K,self.H),self.P)
dt=0.1
F=np.array([[1,dt],[0,1]])
H=np.array([[1,0]])
Q=np.diag([0.1,0.1])
R=np.array([[1]])
x0=np.array([0,0])
P0=np.diag([1,1])
kf=KalmanFi1ter(F,H,Q,R,x0,P0)
truestates=[]
measurements=[]
for i in range(100):
    truestates.append([i*dt,1])
    measurements.append(i*dt+np.random.normal(scale=1))
est_states=[]
for z in measurements:
    kf.predict()
    kf.update(np.array([z]))
    est_states.append(kf.x)
plt.plot([s[0] for s in truestates],label="true")
plt.plot([s[0] for s in est_states],label="Estimate")
plt.legend()
plt.show()
```

### Exp - 6 - Implementation of Semantic Analysis
```py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

f = open("/content/sample_data.txt", "r")
sentences = f.readlines()
f.close()
verbs = [[] for _ in sentences]
i=0
for sentence in sentences:
  print("Sentence",i+1,":", sentence)
  words = word_tokenize(sentence)
  pos_tags = nltk.pos_tag(words)

  for word,tag in pos_tags:
    print(word,"->",tag)

    if tag.startswith('VB'):
      verbs[i].append(word)
  i+=1
  print("\n\n") 

print("Synonyms and Antonymns for verbs in each sentence:\n")
i=0
for sentence in sentences:
  print("Sentence",i+1,":", sentence)
  pos_tags = nltk.pos_tag(verbs[i])
  for word,tag in pos_tags:
    print(word,"->",tag)
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
      for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
          for antonym in lemma.antonyms():
            antonyms.append(antonym.name())

    print("Synonyms:",set(synonyms))
    print("Antonyms:", set(antonyms) if antonyms else "None")
    print()
  print("\n\n")
  i+=1
```

### Exp - 7 - Implementation of Text Summarization
```py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
nltk.download( 'punkt' )
nltk.download( 'stopwords' )

def preprocess_text(text):
	# Tokenize the text into words
	words = word_tokenize(text)
	# Remove stopwords and punctuation
	stop_words= set(stopwords.words( 'english'))
	filtered_words= [word for word in words if word. lower() not in stop_words and word.isalnum()]

	# Stemming
	stemmer = PorterStemmer()
        stemmed_words= [stemmer. stem(word) for word in filtered_words]
	return stemmed_words

def generate_summary(text,num_sentences=3):

	sentences= sent_tokenize(text)
	preprocessed_text = preprocess_text(text)
	# Calculate the frequency of each word
	word_frequencies =nltk. FreqDist (preprocessed_text)

	# Calculate the score for each sentence based on word frequency
	sentence_scores ={}
	for sentence in sentences:
		for word, freq in word_frequencies.items():
			if word in sentence.lower():
				if sentence not in sentence_scores:
					sentence_scores[sentence] = freq
				else:
					sentence_scores[sentence]+= freq
	# Select top N sentences with highest scores
	summary_sentences= sorted(sentence_scores, key=sentence_scores.get,reverse=True) [ : num_sentences]

	return ' '. join(summary_sentences)

input_text = input()
summary = generate_summary(input_text)
print("Origina1 Text: ")
print (input_text )
print( "\nSummary : " )
print(summary)
```

### Exp - 8 - Implementation of Speech Recognition
```py
import speech_recognition as sr
#duration = 25
r = sr.Recognizer()
with sr.Microphone() as source :
    print("Say something:\n\n")
    audio_data = r.listen(source)
try:
    text = r.recognize_google(audio_data)
    print("You said:\n", text)
except sr.UnknownValueError:
    print("Sorry, could not understand audio")
except Exception as e:
    print(f'Error: {e}')
```
