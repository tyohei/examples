#ifndef SHARED_DEMO_H_
#define SHARED_DEMO_H_

void shared_func();

__attribute__((__visibility__("default"))) void default_shared_func();

__attribute__((__visibility__("hidden"))) void hidden_shared_func();

__attribute__((__visibility__("internal"))) void internal_shared_func();

__attribute__((__visibility__("protected"))) void protected_shared_func();

#endif  // SHARED_DEMO_H_

