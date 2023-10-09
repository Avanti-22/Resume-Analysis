from django.db import models
# Create your models here.
class ResumeData(models.Model):
    Resume_No = models.IntegerField(primary_key = True)
    Name = models.CharField(max_length=200)
    Email = models.EmailField("")
    Mobile_No = models.IntegerField()
    Skills = models.TextField(max_length=200)
    Education = models.TextField(max_length=200)
    Experience = models.TextField(max_length=200)

class Matched(models.Model):
    Resume = models.ForeignKey(ResumeData, on_delete=models.CASCADE)
    Extracted_skills = models.TextField()
    Resuired_skills = models.TextField()
    Matched_skills = models.TextField()
    Percent_matched = models.DecimalField(decimal_places=2,max_digits=3)
    
class JobDescription(models.Model):
    Title = models.CharField(max_length=200)
    Required_Skills = models.TextField()
    Requires_Experience = models.CharField(max_length=200)
    
class Contact(models.Model):
    Name=models.CharField(max_length=122)
    Email=models.CharField(max_length=122)
    Subject=models.TextField(max_length=200)
    Message=models.TextField(max_length=300)
    Date=models.DateField() 
    
    def __str__(self):
        return self.Name
    