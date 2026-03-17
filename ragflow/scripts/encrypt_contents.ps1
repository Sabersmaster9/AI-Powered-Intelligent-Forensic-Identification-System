$ErrorActionPreference = 'Stop'
$repo='E:\Users\a\Desktop\ragflow-main\AI-Powered Intelligent Forensic Identification System\ragflow-main'
$src='E:\Users\a\Desktop\ragflow-main\ragflow-main\contents'
Set-Location $repo

if (Test-Path '.\ragflow\contents') { Remove-Item '.\ragflow\contents' -Recurse -Force }
if (Test-Path '.\ragflow\contents.zip') { Remove-Item '.\ragflow\contents.zip' -Force }
if (Test-Path '.\ragflow\contents.encrypted.bin') { Remove-Item '.\ragflow\contents.encrypted.bin' -Force }

Copy-Item -Path $src -Destination '.\ragflow\contents' -Recurse -Force
Compress-Archive -Path '.\ragflow\contents\*' -DestinationPath '.\ragflow\contents.zip' -Force

$plain = [System.IO.File]::ReadAllBytes((Resolve-Path '.\ragflow\contents.zip'))
$aes = [System.Security.Cryptography.Aes]::Create()
$aes.KeySize = 256
$aes.GenerateKey()
$aes.GenerateIV()
$aes.Mode = [System.Security.Cryptography.CipherMode]::CBC
$aes.Padding = [System.Security.Cryptography.PaddingMode]::PKCS7
$enc = $aes.CreateEncryptor()
$cipher = $enc.TransformFinalBlock($plain, 0, $plain.Length)

$outPath = '.\ragflow\contents.encrypted.bin'
$stream = [System.IO.File]::Open($outPath, [System.IO.FileMode]::Create)
try {
  $header = [System.Text.Encoding]::ASCII.GetBytes("RAGFLOW-AES256-CBC`n")
  $stream.Write($header, 0, $header.Length)
  $ivLen = [BitConverter]::GetBytes([int]$aes.IV.Length)
  $stream.Write($ivLen, 0, $ivLen.Length)
  $stream.Write($aes.IV, 0, $aes.IV.Length)
  $stream.Write($cipher, 0, $cipher.Length)
} finally {
  $stream.Dispose()
}

$keyB64 = [Convert]::ToBase64String($aes.Key)
$keyFile='E:\Users\a\Desktop\ragflow-main\AI-Powered Intelligent Forensic Identification System\contents_aes_key.txt'
Set-Content -Path $keyFile -Value $keyB64 -Encoding ASCII

Remove-Item '.\ragflow\contents.zip' -Force
Remove-Item '.\ragflow\contents' -Recurse -Force

Set-Content -Path '.\ragflow\contents.README.txt' -Value @(
'contents folder is encrypted for public repo safety.',
'Encrypted file: contents.encrypted.bin',
'Algorithm: AES-256-CBC',
'Header format: ASCII marker + int32 IV length + IV + ciphertext',
'Decryption key is NOT stored in this repository.'
) -Encoding ASCII

Write-Output 'ENCRYPT_OK'
Write-Output $keyFile
