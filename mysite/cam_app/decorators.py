from functools import wraps
from django.http import HttpResponseForbidden

def same_referer_required(function):
    @wraps(function)
    def wrap(request, *args, **kwargs):
        allowed_referer = '127.0.0.1:8000'
        referer = request.META.get('HTTP_REFERER')
        x_frame_options = request.META.get('HTTP_X_FRAME_OPTIONS')

        print(f"Referer: {referer}")
        print(f"X-Frame-Options: {x_frame_options}")

        if referer is not None and allowed_referer in referer and x_frame_options is None:
            return function(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()
    return wrap
