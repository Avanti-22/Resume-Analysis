# Generated by Django 4.1 on 2023-10-18 16:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0004_rename_resumefile_resumeform_resumefile'),
    ]

    operations = [
        migrations.AlterField(
            model_name='resumeform',
            name='Resumefile',
            field=models.FileField(upload_to=''),
        ),
    ]