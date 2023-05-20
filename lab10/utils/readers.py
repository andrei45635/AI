import pandas as pd


def readReviews(filePath):
    df = pd.read_csv(filePath)
    df.fillna(df.mean(), inplace=True)
    subset = df[['Text', 'Sentiment']]

    text = [subset.iat[i, 0] for i in range(len(subset))]
    sentiment = [subset.iat[i, 1] for i in range(len(subset))]
    labels = list(set(sentiment))

    return text, sentiment, labels


def readEmotions(filePath):
    df = pd.read_csv(filePath)
    df.fillna(df.mean(), inplace=True)
    subset = df[['emailText', 'emailType']]

    emailText = [subset.iat[i, 0] for i in range(len(subset))]
    emailType = [subset.iat[i, 1] for i in range(len(subset))]
    labels = list(set(emailType))

    return emailText, emailType, labels