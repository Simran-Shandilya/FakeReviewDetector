# Generated by Django 4.0 on 2021-12-22 20:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0002_alter_insertreviewmodel_review'),
    ]

    operations = [
        migrations.AlterField(
            model_name='insertreviewmodel',
            name='review',
            field=models.TextField(max_length=1024),
        ),
    ]
