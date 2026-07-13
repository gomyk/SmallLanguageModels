Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue | ForEach-Object {
  '{0}: {1:N1} C' -f $_.InstanceName, (($_.CurrentTemperature/10)-273.15)
}
$cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
'CPU Usage: {0:N1}%' -f $cpu
$os = Get-CimInstance Win32_OperatingSystem
'RAM Used: {0:N1} / {1:N1} GB' -f (($os.TotalVisibleMemorySize-$os.FreePhysicalMemory)/1MB), ($os.TotalVisibleMemorySize/1MB)
