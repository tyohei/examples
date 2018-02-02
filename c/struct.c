#include <stdio.h>


int main(int argc, char **argv) {

  /**
   * Most simple structure.
   *    This does nothing, and we have no way to reference this memory.
   *    Usually this code generates waring since this code has no meaning.
   */
  struct {
    int member_0;
    int member_1;
  };

  /**
   * Simple structure w/o instance declariation.
   *    This creates a Structure named ``tag_name0`` and you can use this as:
   *    
   *      >> struct tag_name0 <name>;
   *      >> struct tag_name0 <name> = {0, 0};
   */
  struct tag_name0 {
    int member_0;
    int member_1;
  };
  struct tag_name0 name_0 = {0, 0};

  /**
   * Simple structure w/ instance declariation.
   *    This creates a Structure named ``tag_name1`` and also declares a
   *    structure instance named ``struct_alias_1``. You can use this structure
   *    as:
   *
   *      >> struct tag_name1 <name>;
   *      >> struct tag_name1 <name> = {0, 0};
   *
   *    Although, it declares a instance ``struct_alias_1``, we can not use
   *    this structure in C normally. We need to **CAST** the initial value
   *    to assighn the declared structure instance.
   */
  struct tag_name1 {
    int member_0;
    int member_1;
  } struct_alias_1;
  // struct_alias_1 = {0, 0};  This is not allowed in C but is allowed C++.
  struct_alias_1 = (struct tag_name1){0, 0};  // This works. Very weird :(.
  struct tag_name1 name_1 = {0, 0};

  /**
   * Typedef structure w/o alias.
   *    This creates a Structure named ``tag_name2`` and use typedef to make
   *    an alias. However, there is no word in the alias place, so we cannot
   *    use the alias, it means this is almost same as no.2 structure. You can
   *    use this structure as:
   *
   *      >> struct tag_name2 <name>;
   *      >> struct tag_name2 <name> = {0, 0};
   *
   *    This code might cause a warning.
   */
  typedef struct tag_name2 {
    int member_0;
    int member_1;
  };
  struct tag_name2 name_2 = {0, 0};

  /**
   * Typedef structure w/ alias.
   *    This creates a Structure named ``tag_name3`` and use typedef to make
   *    an alias. You can use this structure as:
   *
   *      >> structure tag_name3 <name>;
   *      >> structure tag_name3 <name> = {0, 0};
   *      >> struct_alias_3 <name>;
   *      >> struct_alias_3 <name> = {0, 0};
   */
  typedef struct tag_name3 {
    int member_0;
    int member_1;
  } struct_alias_3;
  struct tag_name3 name_30 = {0, 0};
  struct_alias_3 name_31 = {0, 0};

  /**
   * Typedef structure w/ alias w/o tag name.
   *    This creates a Structure without a name and use typedef to make an
   *    alias to this structure. This kind of definition is very usefull.
   *    You can use this structure as:
   *
   *      >> struct_alias_4 <name>;
   *      >> struct_alias_4 <name> = {0, 0};
   */
  typedef struct  {
    int member_0;
    int member_1;
  } struct_alias_4;
  struct_alias_4 name_4 = {0, 0};

  return 0;
}
