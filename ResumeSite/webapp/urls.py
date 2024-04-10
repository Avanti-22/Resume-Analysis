from django.urls import path, include
from django.contrib.auth.views import LoginView,LogoutView
from . import views
# from .views import upload_pdf
from django.conf.urls.static import static
from django.conf import settings
from .views import import_from_excel

# pass for avanti1 is Pass@123
urlpatterns = [
    
    #-----------------------common urls------------------------
    path("", views.home, name="home"),
    path("home/", views.home, name="home"),
    path('aboutus/',views.aboutView,name="aboutus"),
    path('contactus/',views.contactView,name="contactus"),
    path('faq/',views.faqView,name="faq"),
    path('logout/',views.userlogout,name="logout"),
    
    #-----------------------user urls---------------------------
    path('user/',views.userview,name="userview"),   
    path('login/',views.userlogin,name="userlogin"),
    path('upload/',views.upload,name="upload"),
    path('jd_for_matching/',views.jd_for_matching,name="jd_match"),
    path('register/',views.userreg,name="userreg"),
    path('mjob/<int:job_id>/', views.resume_matching, name='resume_matching'),
    
    #-----------------------HR urls-----------------------------
    path('hr/',views.hrview,name="hrview"),
    path('jd_for_ranking/',views.jd_for_ranking,name="jd_rank"),
    path('hr_dash/',views.hr_dash,name="hr_dash"),
    path('hr_jd/',views.hr_jd,name="hr_jd"),
    path('after_new_jd/',views.after_new_jd, name='after_new_jd'),
    path('batch_rjob/<int:job_id>/', views.batch_resume_ranking, name='batch_resume_ranking'),
    path('rjob/<int:job_id>/', views.resume_ranking, name='resume_ranking')
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
