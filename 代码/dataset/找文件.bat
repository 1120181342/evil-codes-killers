@echo off
for /f "delims=" %%a in (1.txt) do (
    copy /y "C:\Users\Administrator\Desktop\samples\APT 29\%%a.zip" "C:\Users\Administrator\Desktop\datset\APT 29"
)

for /f "delims=" %%a in (1.txt) do (
    copy /y "C:\Users\Administrator\Desktop\samples\Dark Hotel\%%a.zip" "C:\Users\Administrator\Desktop\datset\Dark Hotel"
)

for /f "delims=" %%a in (1.txt) do (
    copy /y "C:\Users\Administrator\Desktop\samples\Energetic Bear\%%a.zip" "C:\Users\Administrator\Desktop\datset\Energetic Bear"
)

for /f "delims=" %%a in (1.txt) do (
    copy /y "C:\Users\Administrator\Desktop\samples\APT 28\%%a.zip" "C:\Users\Administrator\Desktop\datset\APT 28"
)