# Generated by Django 4.1.7 on 2023-11-04 14:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0006_alter_resumeform_resumefile'),
    ]

    operations = [
        migrations.AlterField(
            model_name='resumeform',
            name='Resumefile',
            field=models.FileField(default='', upload_to='pdfs/'),
        ),
    ]