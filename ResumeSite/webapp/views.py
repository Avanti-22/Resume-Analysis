from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, logout, login
from datetime import datetime
from django.contrib import messages
from webapp.models import Contact, Resumeform, JobDescription, ResumeData
import os
from .models import ResumeData,Matched
from django.db import models
from django.shortcuts import render
import requests
from openpyxl import load_workbook
from bs4 import BeautifulSoup

#from .models import Details

from .matching import fetch_data

# from .forms import PDFUploadForm

# Create your views here.
def home(request):
    return render(request,'home.html')
# @login_required()

def aboutView(request):
    return render(request,'aboutus.html')

def contactView(request):
    if request.method =="POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        contact = Contact(Name=name, Email=email, Subject=subject, Message=message, Date=datetime.today())
        contact.save()
        messages.success(request, "Your message has been sent.")
    return render(request,'contactus.html')

def faqView(request):
    return render(request,'faq.html')

# def dashboardView(request):
#     return render(request,'dashboard.html')

def hrview(request):
    
    if request.method=="POST":
        username=request.POST.get('username')
        password=request.POST.get('password')
        #validate the user
        print(username,password)

        user = authenticate(username=username, password=password)
        if user is not None:
            # A backend authenticated the credentials
            login(request, user)
            return render(request,'req_form.html')    
        else:
            # No backend authenticated the credentials
            return render(request, 'hrview.html')

    return render(request, 'hrview.html')

def userview(request):
    return render(request, 'candidateview.html')

def userlogin(request):
    if request.method=="POST":
        username=request.POST.get('username')
        password=request.POST.get('password')
        #validate the user
        print(username,password)

        user = authenticate(username=username, password=password)
        if user is not None:
            # A backend authenticated the credentials
            login(request, user)
            return render(request,'upload.html')    
        else:
            # No backend authenticated the credentials
            return render(request, 'login.html')

    return render(request, 'login.html')



def upload(request):
    if request.user.is_anonymous:
        return render(request, 'login.html')

    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        resumefile = request.FILES.get('file')  # Use request.FILES to access the uploaded file

        if resumefile:
            # Create the directory if it doesn't exist
            upload_folder = 'pdfs'
            os.makedirs(upload_folder, exist_ok=True)

            # Save the uploaded file to the specified location
            with open(os.path.join(upload_folder, resumefile.name), 'wb') as destination:
                for chunk in resumefile.chunks():
                    destination.write(chunk)

            # Create and save the model instance
            resumeform = Resumeform(Name=name, Email=email, Resumefile=os.path.join(upload_folder, resumefile.name))
            resumeform.save()
            resume_id = resumeform.id
            request.session['resume_id'] = resume_id
            messages.success(request, "Your form has been submitted.")
            return render(request, 'jobdescription.html')  # Redirect to a success page
    else:
        # Handle GET request or render the initial form
        return render(request, 'upload.html')

    # return render(request, 'upload.html')

def req(request):
    if request.user.is_anonymous:
        return render(request,'login.html')
    return render(request, 'ref_form.html')

def userlogout(request):
    logout(request)
    return render(request,'home.html')

def userreg(request):
    if request.method == 'POST':
        # name = request.POST['name']
        email = request.POST.get('email')
        username = request.POST.get('username')
        password = request.POST.get('password')
        cpassword = request.POST.get('cpassword')
        print(email,username,password,cpassword)
        if password == cpassword:
            if User.objects.filter(username=username).exists():
                messages.info(
                    request, 'This username already exists. Please try again!!')
                return render(request, 'register.html')
            else:
                user = User(
                    email=email, username=username, password=password)
                user.set_password(password)
                user.save()
                
                messages.info(
                    request, 'Account created successfully.')
                return render(request,'login.html')
        else:
            messages.info(
                request, 'The Passwords doesnot match. Please try again!!')
            return render(request, 'register.html')

    return render(request, 'register.html')

def jd(request):
    
    return render(request,'jobdescription.html')

def resume_matching(request, job_id):
    resume_id = request.session.get('resume_id')
    resume = get_object_or_404(Resumeform, id=resume_id)
    job_description = get_object_or_404(JobDescription, id=job_id)

        # Assuming fetch_data returns a tuple
    resume_content = fetch_data(resume_id, job_id)

    # Accessing tuple elements directly
    email = resume_content[1]  # Assuming email is at index 1
    mobile_number = resume_content[2]  # Assuming mobile_number is at index 2
    skills = resume_content[3]  # Assuming skills is at index 5

    # Creating ResumeData object
    resume_data = ResumeData.objects.create(
        Email=email,
        Mobile_No=mobile_number,
        Skills=skills,
        # Add other fields as needed
    )
    print("matched percent is",resume_content[0])
    match_percent =round(resume_content[0], 2)  # Assuming match_percent is at index 0
    
    return render(request, 'matching.html', {'resume': resume,'mobile_number':mobile_number,'email':email,'skills':skills,'job_description': job_description, 'match_percent': match_percent})

def resume_ranking(request):
    # Calculate match percentage for each resume and order them by match percentage
    resumes = ResumeData.objects.all()
    ranked_resumes = sorted(resumes, key=lambda resume: resume.matched_set.all().aggregate(avg_match=models.Avg('Percent_matched'))['avg_match'] or 0, reverse=True)

    # Render template with ranked resumes
    return render(request, 'resume_ranking.html', {'resumes': ranked_resumes})

def import_from_excel(request):
    if request.method == 'POST':
        excel_file = request.FILES[r"ResumeProcessing\Extracted Data\Extracted.xlsx"]
        wb = load_workbook(excel_file)
        ws = wb.active

        for row in ws.iter_rows(min_row=2, values_only=True):
            name, email, phone_number, skills = row
            Details.objects.create(name=name, email=email, phone_number=phone_number, skills=skills)

        return render(request, 'import_success.html')

    return render(request, 'import_form.html')

def display_data(request):
    data = Details.objects.all()
    return render(request, 'display_data.html', {'data': data})

def remote_jobs_view(request):
    url = 'https://remotive.com/api/remote-jobs'
    response = requests.get(url)
    data = response.json()
    jobs = data.get('jobs', [])

    # Remove HTML tags from job descriptions
    for job in jobs:
        html_content = job.get('description', '')
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        # Get text without HTML tags
        text_without_html = soup.get_text(separator=' ')
        # Update job description with text without HTML tags
        job['description'] = text_without_html.strip()

    return render(request, 'jobs.html', {'jobs': jobs})

# def upload_pdf(request):
#     if request.method == 'POST':
#         form = PDFUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             pdf_file = form.save()
#             return redirect('success_url')  # Redirect to a success page

#     else:
#         form = PDFUploadForm()

#     return render(request, 'upload_pdf.html', {'form': form})

# def registerView(request):
#     if request.method == "POST":
#         form = UserCreationForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('login_url')
#     else:
#         form = UserCreationForm()
#     return render(request,'registration/register.html',{'form':form})