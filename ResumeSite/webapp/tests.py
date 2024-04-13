from django.test import TestCase

from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import Resumeform, JobDescription, Contact,ResumeData
from django.core.exceptions import ValidationError
from datetime import date
from django.urls import reverse
from django.contrib.auth.models import User



#--------------------TESTCASES FOR MODELS-----------------------
# Django TestCases for the Resumeform model
class ResumeformTestCase(TestCase):

    def setUp(self):
        # Create a sample resume file for testing
        self.resume_file = SimpleUploadedFile("resume.pdf", b"file_content", content_type="application/pdf")

    def test_resumeform_creation(self):
        # Test creating a Resumeform instance
        resumeform = Resumeform.objects.create(
            Name="John Doe",
            Email="john@example.com",
            Resumefile=self.resume_file,
            # You can add ResumeId here if needed
        )

        # Assert that the instance is created successfully
        self.assertEqual(resumeform.Name, "John Doe")
        self.assertEqual(resumeform.Email, "john@example.com")
        self.assertEqual(resumeform.Resumefile.name, "pdfs/resume.pdf")

class JobDescriptionTestCase(TestCase):

    def test_jobdescription_creation(self):
        # Test creating a JobDescription instance
        job_description = JobDescription.objects.create(
            Title="Software Engineer",
            Required_Skills="Python, Django",
            Requires_Experience="3 years",
        )

        # Assert that the instance is created successfully
        self.assertEqual(job_description.Title, "Software Engineer")
        self.assertEqual(job_description.Required_Skills, "Python, Django")
        self.assertEqual(job_description.Requires_Experience, "3 years")

# Add more test cases as needed for JobDescription model
class ContactTestCase(TestCase):

    def test_contact_creation(self):
        # Test creating a Contact instance
        contact = Contact.objects.create(
            Name="John Doe",
            Email="john@example.com",
            Subject="Inquiry",
            Message="This is a test message.",
            Date=date.today(),
        )

        # Assert that the instance is created successfully
        self.assertEqual(contact.Name, "John Doe")
        self.assertEqual(contact.Email, "john@example.com")
        self.assertEqual(contact.Subject, "Inquiry")
        self.assertEqual(contact.Message, "This is a test message.")
        self.assertEqual(contact.Date, date.today())


#--------------------------TESTCASES FOR URLS---------------------
# Django TestCases for the urlpatterns in your Django project
class UrlsTestCase(TestCase):
    def setUp(self):
        # Create a test user
        self.user = User.objects.create_user(username='testuser', password='testpass')

    def test_home_url(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)

    def test_aboutus_url(self):
        response = self.client.get(reverse('aboutus'))
        self.assertEqual(response.status_code, 200)

    def test_login_url(self):
        response = self.client.get(reverse('userlogin'))
        self.assertEqual(response.status_code, 200)

    def test_upload_url(self):
        # Log in the test user before accessing the upload URL
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('upload'))
        self.assertEqual(response.status_code, 200)

    def test_resume_ranking_url(self):
        response = self.client.get(reverse('resume_ranking'))
        self.assertEqual(response.status_code, 200)

    def test_resume_matching_url(self):
        # Create a JobDescription instance for testing
        job_description = JobDescription.objects.create(Title="Test Job", Required_Skills="Python", Requires_Experience="2 years")
        response = self.client.get(reverse('resume_matching', args=[job_description.id]))
        self.assertEqual(response.status_code, 200)

#----------------------TESTCASES FOR VIEWS----------------------------------
class ViewsTestCase(TestCase):
    def setUp(self):
        # Create a test user for authentication tests
        self.user = User.objects.create_user(username='testuser', password='testpass')

        # Create other necessary test objects as needed

    def test_home_view(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)

    def test_about_view(self):
        response = self.client.get(reverse('aboutus'))
        self.assertEqual(response.status_code, 200)

    def test_contact_view(self):
        response = self.client.get(reverse('contactus'))
        self.assertEqual(response.status_code, 200)

    def test_contact_view_post(self):
        response = self.client.post(reverse('contactus'), {'name': 'Test', 'email': 'test@example.com', 'subject': 'Test Subject', 'message': 'Test Message'})
        self.assertEqual(response.status_code, 200)

        # Add assertions based on your logic for a successful message submission

    def test_hr_view(self):
        response = self.client.get(reverse('hrview'))
        self.assertEqual(response.status_code, 200)

    # Add more test cases for other views

    def test_upload_view_unauthenticated(self):
        response = self.client.get(reverse('upload'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'login.html')  # Assuming login.html is the template for unauthenticated users

    def test_upload_view_authenticated(self):
        # Log in the test user before accessing the upload view
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(reverse('upload'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateNotUsed(response, 'login.html')  # Assuming login.html is not used for authenticated users

    # Add more test cases as needed

