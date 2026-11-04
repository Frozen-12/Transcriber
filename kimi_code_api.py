"""Portable Kimi Code Chat Completions core. Import does not perform I/O.

Host supplies private config; this module never persists or prints credentials.
Requires jsonschema. request injection lets tests avoid all network access.
"""
import json
import re
import socket
import urllib.error
import urllib.request

ENDPOINTS={'global':'https://api.kimi.ai/coding/v1','mainland-cn':'https://api.kimi.com/coding/v1'}
EFFORTS={m:('low','high','max') for m in ('k3.1-pro','k3.1-pro-highspeed','k3','k3-256k','kimi-for-coding')}
EFFORTS['kimi-for-coding-highspeed']=()

def valid_key(key):return isinstance(key,str) and 16<=len(key)<=4096 and all(33<=ord(c)<=126 for c in key)

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None

def http_json(region,key,path,payload=None,timeout=60):
 if region not in ENDPOINTS or path not in ('/models','/chat/completions'):raise ValueError('KIMI_API_INVALID_ENDPOINT')
 if not valid_key(key):raise ValueError('KIMI_API_INVALID_KEY')
 if type(timeout) not in (int,float) or not 1<=timeout<=900:raise ValueError('KIMI_API_INVALID_TIMEOUT')
 req=urllib.request.Request(ENDPOINTS[region]+path,data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None,
  headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'Kimi-Code-API-Integration'})
 try:
  with urllib.request.build_opener(NoRedirect()).open(req,timeout=timeout) as response:raw=response.read(4_000_001)
  if len(raw)>4_000_000:raise ValueError('KIMI_API_RESPONSE_TOO_LARGE')
  value=json.loads(raw)
  if not isinstance(value,dict):raise ValueError('KIMI_API_INVALID_RESPONSE')
  return value
 except urllib.error.HTTPError as error:
  code={401:'AUTH_REQUIRED',403:'ACCESS_DENIED',429:'USAGE_LIMIT',400:'BAD_REQUEST'}.get(error.code,'HTTP_ERROR')
  raise ValueError('KIMI_API_'+code) from None
 except (TimeoutError,socket.timeout):raise ValueError('KIMI_API_TIMEOUT') from None
 except urllib.error.URLError:raise ValueError('KIMI_API_UNAVAILABLE') from None
 except (json.JSONDecodeError,UnicodeError):raise ValueError('KIMI_API_INVALID_RESPONSE') from None

def discover_models(region,key,request=None):
 if region not in ENDPOINTS or not valid_key(key):raise ValueError('KIMI_API_INVALID_CONFIG')
 reply=(request or http_json)(region,key,'/models')
 data=reply.get('data')
 if not isinstance(data,list):raise ValueError('KIMI_API_INVALID_RESPONSE')
 models=list(dict.fromkeys(m['id'] for m in data if isinstance(m,dict) and isinstance(m.get('id'),str) and m['id'] in EFFORTS))
 if not models:raise ValueError('KIMI_API_MODELS_UNAVAILABLE')
 return models

def complete_json(config,text,schema,model,effort='default',timeout=60,request=None):
 from jsonschema import Draft202012Validator
 if not isinstance(config,dict) or config.get('region') not in ENDPOINTS or not valid_key(config.get('api_key')):raise ValueError('KIMI_API_INVALID_CONFIG')
 if not isinstance(model,str) or model not in EFFORTS or model not in config.get('models',[]):raise ValueError('KIMI_API_MODEL_UNAVAILABLE')
 if effort!='default' and effort not in EFFORTS[model]:raise ValueError('KIMI_API_UNSUPPORTED_EFFORT')
 limit=config.get('max_output_tokens',8192)
 if type(limit)!=int or not 1024<=limit<=32768:raise ValueError('KIMI_API_INVALID_OUTPUT_LIMIT')
 if type(timeout) not in (int,float) or not 1<=timeout<=900:raise ValueError('KIMI_API_INVALID_TIMEOUT')
 if not isinstance(text,str) or not text.strip():raise ValueError('KIMI_API_EMPTY_PROMPT')
 try:Draft202012Validator.check_schema(schema)
 except Exception:raise ValueError('KIMI_API_INVALID_SCHEMA') from None
 prompt=text+'\n\nReturn only JSON matching this schema; no tools or commentary:\n'+json.dumps(schema,ensure_ascii=False,separators=(',',':'))
 payload={'model':model,'messages':[{'role':'user','content':prompt}],'stream':False,'max_tokens':limit,'thinking':{'type':'enabled'}}
 if effort!='default':payload['reasoning_effort']=effort
 reply=(request or http_json)(config['region'],config['api_key'],'/chat/completions',payload,timeout=timeout)
 if not isinstance(reply,dict):raise ValueError('KIMI_API_INVALID_RESPONSE')
 choices=reply.get('choices')
 if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):raise ValueError('KIMI_API_INVALID_RESPONSE')
 choice=choices[0];message=choice.get('message')
 if not isinstance(message,dict):raise ValueError('KIMI_API_INVALID_RESPONSE')
 if message.get('tool_calls') or message.get('function_call'):raise ValueError('KIMI_API_UNEXPECTED_TOOL_CALL')
 if choice.get('finish_reason')!='stop':raise ValueError('KIMI_API_INCOMPLETE_RESPONSE')
 content=message.get('content')
 if not isinstance(content,str):raise ValueError('KIMI_API_INVALID_RESPONSE')
 fence=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',content.strip(),re.S)
 try:value=json.loads(fence[1] if fence else content)
 except json.JSONDecodeError:raise ValueError('KIMI_API_INVALID_JSON') from None
 try:Draft202012Validator(schema).validate(value)
 except Exception:raise ValueError('KIMI_API_SCHEMA_MISMATCH') from None
 raw=reply.get('usage');usage={}
 if isinstance(raw,dict):
  for out,field in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
   if type(raw.get(field))==int and raw[field]>=0:usage[out]=raw[field]
  for out,detail,field in [('cached_input_tokens','prompt_tokens_details','cached_tokens'),('thinking_tokens','completion_tokens_details','reasoning_tokens')]:
   details=raw.get(detail);v=details.get(field) if isinstance(details,dict) else None
   if type(v)==int and v>=0:usage[out]=v
 actual=reply.get('model');actual=actual if isinstance(actual,str) and re.fullmatch(r'[A-Za-z0-9._/-]{1,120}',actual) else None
 return value,usage or None,actual
