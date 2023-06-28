# IPASS

Tekstclassificatie met Meerdere Labels


Dit project bevat een implementatie van een logistisch regressiemodel voor tekstclassificatie met meerdere labels. Het model is getraind om tekstuele gegevens te analyseren en deze te categoriseren in verschillende vooraf gedefinieerde labels. Dit README-bestand geeft een overzicht van het project en bevat instructies om het model te gebruiken en te reproduceren.


Inhoudsopgave

1. Projectdoel
2. Projectstructuur
3. Installatie
4. Gebruik
5. Licentie

Projectdoel

Het doel van dit project is om een logistisch regressiemodel te ontwikkelen dat tekstuele gegevens kan classificeren in meerdere labels. Het model maakt gebruik van machine learning en natuurlijke taalverwerkingstechnieken om tekst te analyseren en te categoriseren. Het kan worden toegepast in verschillende domeinen zoals sentimentanalyse, spamdetectie en inhoudsclassificatie op basis van specifieke labels.


Projectstructuur

De projectstructuur is als volgt opgezet:

- README.md: Dit bestand bevat een overzicht van het project en de instructies.
- chat.py: Het hoofdbestand van het project dat de tekstclassificatie-functionaliteit implementeert.
- database.py: Bevat code voor het verbinden met en het beheren van de database.
- test.py: Een script dat gebruikt kan worden om het getrainde model te testen.
- training_set.py:Traint de computer.
- profane_words.txt: Een tekstbestand met woorden die als aanstootgevend worden beschouwd.
- pipeline.pkl: Een opgeslagen object van het trainingsproces.
- identity_hate_model.pkl: Een opgeslagen model voor het label 'identity_hate'.
- insult_model.pkl: Een opgeslagen model voor het label 'insult'.
- obscene_model.pkl: Een opgeslagen model voor het label 'obscene'.
- severe_toxic_model.pkl: Een opgeslagen model voor het label 'severe_toxic'.
- threat_model.pkl: Een opgeslagen model voor het label 'threat'.
- toxic_model.pkl: Een opgeslagen model voor het label 'toxic'.
- train.csv: De trainingsdataset in CSV-formaat.
- test.csv: De testdataset in CSV-formaat.
- SQL script for messages.sql: Een SQL-script voor het maken van de berichtendatabase.

Installatie

1. clone de repository
2. zorg ervoor dat alle imports zijn gedownload
	- pandas
	- scikit-learn
	- nltk
	- re
	- swifter
	- joblib


Gebruik

Volg deze stappen om het model te gebruiken en te testen:

1. Zorg ervoor dat je de database hebt opgezet met behulp van het meegeleverde SQL-script.
2. Voer het trainingsproces uit om het model te trainen:
3. Voer het testscript uit om het getrainde model te testen:
4. Met de chat.py kan er in de commandline worden ingelogd en commentaar worden achtergelaten.
5. Om chat.py te kunnen testen kan je met deze gegevens inloggen: 
```
example : StarGazer92
example : stargazer92@example.com
example : 4352
```




Licentie

Dit project is gelicentieerd onder de MIT-licentie. Je bent vrij om de broncode te gebruiken, aan te passen en te distribueren volgens de voorwaarden van de licentie.

Bronnen:
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python
https://towardsdatascience.com/hybrid-rule-based-machine-learning-with-scikit-learn-9cb9841bebf2
chatgpt
