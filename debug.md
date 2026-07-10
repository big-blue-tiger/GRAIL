# trans to 9991
```bash
rsync -aP --checksum \
  --no-owner --no-group --omit-dir-times \
  -e "ssh -p 9991 -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  /home/tide/robot/GRAIL/ \
  ygc@202.120.37.249:/home/ygc/data0/GRAIL/

rsync -aP --checksum \
  --no-owner --no-group --omit-dir-times \
  -e "ssh -p 9991 -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  /home/tide/robot/Hunyuan3D-2.1/ \
  ygc@202.120.37.249:/home/ygc/data0/Hunyuan3D-2.1/
```

# api
apiKey
sk-ws-H.EMLEMMI.PdPj.MEYCIQCRofma6q8VafElwlbZ6AFpZsTexnVpDhBQ1g-LHQVsVwIhAJsUtXr5SZJ6L7IhkpQhT4CgIfNYO9c3n8CkFmkJxTbb
apiHost
ws-ckaaiw57ig8d28nh.cn-beijing.maas.aliyuncs.com
openAiCompatible
https://ws-ckaaiw57ig8d28nh.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
dashScope
https://ws-ckaaiw57ig8d28nh.cn-beijing.maas.aliyuncs.com/api/v1
description
hunyuan3d
workspaceName
默认业务空间
workspaceId
ws-ckaaiw57ig8d28nh
```bash
export DASHSCOPE_API_KEY="sk-ws-H.EMLEMMI.PdPj.MEYCIQCRofma6q8VafElwlbZ6AFpZsTexnVpDhBQ1g-LHQVsVwIhAJsUtXr5SZJ6L7IhkpQhT4CgIfNYO9c3n8CkFmkJxTbb"
export DASHSCOPE_WORKSPACE_ID="ws-ckaaiw57ig8d28nh"

```
# create hunyuan docker
```bash
nvidia-docker run -it --name 'hunyuan3d' --gpus all --shm-size 32g -e NVIDIA_DRIVER_CAPABILITIES=all -v /data0/dataset/:/data -v /data0/ygc/:/home pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel /bin/bash


```  