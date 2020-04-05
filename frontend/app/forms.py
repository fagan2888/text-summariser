from flask_wtf import FlaskForm
from wtforms import TextAreaField, PasswordField, SubmitField
from wtforms.validators import DataRequired

class TextSumForm(FlaskForm):
    inputText = TextAreaField('Article to summarize', validators=[DataRequired()])
  #  outputText = TextAreaField('Summary', cols=60, rows=20)
    submit = SubmitField('Summarize')