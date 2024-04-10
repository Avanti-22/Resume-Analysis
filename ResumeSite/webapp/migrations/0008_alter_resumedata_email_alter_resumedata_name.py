# Generated by Django 5.0.4 on 2024-04-09 11:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0007_rename_email_resumeform_r_email_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='resumedata',
            name='Email',
            field=models.EmailField(blank=True, default='default@mail.com', max_length=254, null=True),
        ),
        migrations.AlterField(
            model_name='resumedata',
            name='Name',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]