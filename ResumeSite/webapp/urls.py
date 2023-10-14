from django.urls import path, include
from django.contrib.auth.views import LoginView,LogoutView
from . import views

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
    path('req/',views.req,name="req"),
    path('register/',views.userreg,name="userreg"),
    path('logout/',views.userlogout,name="logout"),
]