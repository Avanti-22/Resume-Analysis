# Generated by Django 4.1 on 2023-10-18 14:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0002_contact'),
    ]

    operations = [
        migrations.CreateModel(
            name='Resumeform',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Name', models.CharField(max_length=122)),
                ('Email', models.CharField(max_length=122)),
                ('resumefile', models.FileField(upload_to='pdfs/')),
            ],
        ),
    ]