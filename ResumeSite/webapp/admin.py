from django.contrib import admin

# Register your models here.
from .models import ResumeData
from .models import Resumeform
from .models import Contact
from .models import JobDescription
from .models import Matched


admin.site.register(ResumeData)
admin.site.register(Resumeform)
admin.site.register(Contact)
admin.site.register(JobDescription)
admin.site.register(Matched)