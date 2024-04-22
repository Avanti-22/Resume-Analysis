from django.db import models
# Create your models here.
   
class Resumeform(models.Model):
    Resume_No = models.AutoField(primary_key = True)
    R_Name=models.CharField(max_length=122,default='No Name Found')
    R_Email=models.CharField(max_length=122,default='default@mail.com')
    Resumefile = models.FileField(upload_to='pdfs/',default='')
    # def __str__(self):
    #     return str(self.Resume_No)


class JobDescription(models.Model):
    Title = models.CharField(max_length=200)
    Description =models.TextField(max_length=1000, default=' ')
    Required_Skills = models.TextField()
    Required_Experience = models.CharField(max_length=200)
    Job_id= models.AutoField(primary_key= True)
    # def __str__(self):
    #     return str(self.Job_id) 
    
class ResumeData(models.Model):
    Resume = models.ForeignKey(Resumeform, on_delete=models.CASCADE)  # Foreign key
    Name = models.CharField(max_length=200, null=True, blank=True)
    Email = models.EmailField(default='default@mail.com',null=True, blank=True)
    Mobile_No = models.IntegerField(default=9999999999, null=True, blank=True)
    Skills = models.TextField(max_length=200, default='No skills found')
    Education = models.TextField(max_length=200, default='No education found')
    Experience = models.TextField(max_length=200, default='No experience found')
     
    
class Matched(models.Model):
    Resume = models.ForeignKey(Resumeform, on_delete=models.CASCADE)  
    Job= models.ForeignKey(JobDescription, on_delete=models.CASCADE)
    Extracted_skills = models.TextField(default='No skills extracted')
    Required_skills = models.TextField(default='No required skills provided')
    Matched_skills = models.TextField(default='No matched skills')
    Percent_matched = models.DecimalField(decimal_places=2,max_digits=5,default=0.00)
    Name = models.CharField(max_length=200, null=True, blank=True)
    Email = models.EmailField(default='default@mail.com',null=True, blank=True)
    Mobile_No = models.IntegerField(default=9999999999, null=True, blank=True)
    
class Contact(models.Model):
    Name=models.CharField(max_length=122)
    Email=models.CharField(max_length=122)
    Subject=models.TextField(max_length=200)
    Message=models.TextField(max_length=300)
    Date=models.DateField()

    # def __str__(self):
    #     return self.Name

    def __str__(self):
        return self.Name 
    # upload_date = models.DateTimeField(auto_now_add=True)

        

