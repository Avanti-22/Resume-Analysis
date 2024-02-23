from django.urls import path, include
from django.contrib.auth.views import LoginView,LogoutView
from . import views
# from .views import upload_pdf
from django.conf.urls.static import static
from django.conf import settings

# pass for avanti1 is Pass@123
urlpatterns = [
    path("", views.home, name="home"),
    path("home/", views.home, name="home"),
    # path('dashboard/',views.dashboardView,name="dashboard"),
    path('aboutus/',views.aboutView,name="aboutus"),
    path('contactus/',views.contactView,name="contactus"),
    path('faq/',views.faqView,name="faq"), 
    path('hr/',views.hrview,name="hrview"),
    path('user/',views.userview,name="userview"),   
    path('login/',views.userlogin,name="userlogin"),
    path('upload/',views.upload,name="upload"),
    path('jd/',views.jd,name="jd"),
    path('req/',views.req,name="req"),
    path('register/',views.userreg,name="userreg"),
    path('logout/',views.userlogout,name="logout"),
    # matching_app/urls.py
    path('resume_ranking/', views.resume_ranking, name='resume_ranking'),
    path('job/<int:job_id>/', views.resume_matching, name='resume_matching')

    # path('upload/', upload_pdf, name='upload_pdf'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
