from __future__ import absolute_import

from django.conf import settings
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.exceptions import ObjectDoesNotExist
from django.core.validators import RegexValidator, ip_address_validators
from django.forms import FilePathField as DjangoFilePathField
from django.forms import ImageField as DjangoImageField
from django.utils import six, timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.encoding import is_protected_type, smart_text
from django.utils.functional import cached_property
from django.utils.ipv6 import clean_ipv6_address

from rest_framework import permissions
from rest_framework.serializers import ModelSerializer, ValidationError
from rest_framework.viewsets import ModelViewSet

from push_notifications.models import APNSDevice, GCMDevice
from push_notifications.fields import hex_re

from django.utils.translation import ugettext_lazy as _

import re


class empty:
    """
    This class is used to represent no data being provided for a given input
    or output value.
    It is required because `None` may be a valid input or output value.
    """
    pass


def is_simple_callable(obj):
    """
    True if the object is a callable that takes no arguments.
    """
    function = inspect.isfunction(obj)
    method = inspect.ismethod(obj)

    if not (function or method):
        return False

    args, _, _, defaults = inspect.getargspec(obj)
    len_args = len(args) if function else len(args) - 1
    len_defaults = len(defaults) if defaults else 0
    return len_args <= len_defaults


def get_attribute(instance, attrs):
    """
    Similar to Python's built in `getattr(instance, attr)`,
    but takes a list of nested attributes, instead of a single attribute.
    Also accepts either attribute lookup on objects or dictionary lookups.
    """
    for attr in attrs:
        if instance is None:
            # Break out early if we get `None` at any point in a nested lookup.
            return None
        try:
            if isinstance(instance, collections.Mapping):
                instance = instance[attr]
            else:
                instance = getattr(instance, attr)
        except ObjectDoesNotExist:
            return None
        if is_simple_callable(instance):
            try:
                instance = instance()
            except (AttributeError, KeyError) as exc:
                # If we raised an Attribute or KeyError here it'd get treated
                # as an omitted field in `Field.get_attribute()`. Instead we
                # raise a ValueError to ensure the exception is not masked.
                raise ValueError(
                    'Exception raised in callable attribute "{0}"; original exception was: {1}'.format(attr, exc))

    return instance


def set_value(dictionary, keys, value):
    """
    Similar to Python's built in `dictionary[key] = value`,
    but takes a list of nested keys instead of a single key.
    set_value({'a': 1}, [], {'b': 2}) -> {'a': 1, 'b': 2}
    set_value({'a': 1}, ['x'], 2) -> {'a': 1, 'x': 2}
    set_value({'a': 1}, ['x', 'y'], 2) -> {'a': 1, 'x': {'y': 2}}
    """
    if not keys:
        dictionary.update(value)
        return

    for key in keys[:-1]:
        if key not in dictionary:
            dictionary[key] = {}
        dictionary = dictionary[key]

    dictionary[keys[-1]] = value


def to_choices_dict(choices):
    """
    Convert choices into key/value dicts.
    pairwise_choices([1]) -> {1: 1}
    pairwise_choices([(1, '1st'), (2, '2nd')]) -> {1: '1st', 2: '2nd'}
    pairwise_choices([('Group', ((1, '1st'), 2))]) -> {'Group': {1: '1st', 2: '2nd'}}
    """
    # Allow single, paired or grouped choices style:
    # choices = [1, 2, 3]
    # choices = [(1, 'First'), (2, 'Second'), (3, 'Third')]
    # choices = [('Category', ((1, 'First'), (2, 'Second'))), (3, 'Third')]
    ret = OrderedDict()
    for choice in choices:
        if (not isinstance(choice, (list, tuple))):
            # single choice
            ret[choice] = choice
        else:
            key, value = choice
            if isinstance(value, (list, tuple)):
                # grouped choices (category, sub choices)
                ret[key] = to_choices_dict(value)
            else:
                # paired choice (key, display value)
                ret[key] = value
    return ret


def flatten_choices_dict(choices):
    """
    Convert a group choices dict into a flat dict of choices.
    flatten_choices({1: '1st', 2: '2nd'}) -> {1: '1st', 2: '2nd'}
    flatten_choices({'Group': {1: '1st', 2: '2nd'}}) -> {1: '1st', 2: '2nd'}
    """
    ret = OrderedDict()
    for key, value in choices.items():
        if isinstance(value, dict):
            # grouped choices (category, sub choices)
            for sub_key, sub_value in value.items():
                ret[sub_key] = sub_value
        else:
            # choice (key, display value)
            ret[key] = value
    return ret


def iter_options(grouped_choices, cutoff=None, cutoff_text=None):
    """
    Helper function for options and option groups in templates.
    """

    class StartOptionGroup(object):
        start_option_group = True
        end_option_group = False

        def __init__(self, label):
            self.label = label

    class EndOptionGroup(object):
        start_option_group = False
        end_option_group = True

    class Option(object):
        start_option_group = False
        end_option_group = False

        def __init__(self, value, display_text, disabled=False):
            self.value = value
            self.display_text = display_text
            self.disabled = disabled

    count = 0

    for key, value in grouped_choices.items():
        if cutoff and count >= cutoff:
            break

        if isinstance(value, dict):
            yield StartOptionGroup(label=key)
            for sub_key, sub_value in value.items():
                if cutoff and count >= cutoff:
                    break
                yield Option(value=sub_key, display_text=sub_value)
                count += 1
            yield EndOptionGroup()
        else:
            yield Option(value=key, display_text=value)
            count += 1

    if cutoff and count >= cutoff and cutoff_text:
        cutoff_text = cutoff_text.format(count=cutoff)
        yield Option(value='n/a', display_text=cutoff_text, disabled=True)


class CreateOnlyDefault(object):
    """
    This class may be used to provide default values that are only used
    for create operations, but that do not return any value for update
    operations.
    """

    def __init__(self, default):
        self.default = default

    def set_context(self, serializer_field):
        self.is_update = serializer_field.parent.instance is not None
        if callable(self.default) and hasattr(self.default, 'set_context') and not self.is_update:
            self.default.set_context(serializer_field)

    def __call__(self):
        if self.is_update:
            raise SkipField()
        if callable(self.default):
            return self.default()
        return self.default

    def __repr__(self):
        return unicode_to_repr(
            '%s(%s)' % (self.__class__.__name__, unicode_repr(self.default))
        )


class CurrentUserDefault(object):
    def set_context(self, serializer_field):
        self.user = serializer_field.context['request'].user

    def __call__(self):
        return self.user

    def __repr__(self):
        return unicode_to_repr('%s()' % self.__class__.__name__)


class SkipField(Exception):
    pass


NOT_READ_ONLY_WRITE_ONLY = 'May not set both `read_only` and `write_only`'
NOT_READ_ONLY_REQUIRED = 'May not set both `read_only` and `required`'
NOT_REQUIRED_DEFAULT = 'May not set both `required` and `default`'
USE_READONLYFIELD = 'Field(read_only=True) should be ReadOnlyField'
MISSING_ERROR_MESSAGE = (
    'ValidationError raised by `{class_name}`, but error key `{key}` does '
    'not exist in the `error_messages` dictionary.'
)


class Field(object):
    _creation_counter = 0

    default_error_messages = {
        'required': _('This field is required.'),
        'null': _('This field may not be null.')
    }
    default_validators = []
    default_empty_html = empty
    initial = None

    def __init__(self, read_only=False, write_only=False,
                 required=None, default=empty, initial=empty, source=None,
                 label=None, help_text=None, style=None,
                 error_messages=None, validators=None, allow_null=False):
        self._creation_counter = Field._creation_counter
        Field._creation_counter += 1

        # If `required` is unset, then use `True` unless a default is provided.
        if required is None:
            required = default is empty and not read_only

        # Some combinations of keyword arguments do not make sense.
        assert not (read_only and write_only), NOT_READ_ONLY_WRITE_ONLY
        assert not (read_only and required), NOT_READ_ONLY_REQUIRED
        assert not (required and default is not empty), NOT_REQUIRED_DEFAULT
        assert not (read_only and self.__class__ == Field), USE_READONLYFIELD

        self.read_only = read_only
        self.write_only = write_only
        self.required = required
        self.default = default
        self.source = source
        self.initial = self.initial if (initial is empty) else initial
        self.label = label
        self.help_text = help_text
        self.style = {} if style is None else style
        self.allow_null = allow_null

        if self.default_empty_html is not empty:
            if default is not empty:
                self.default_empty_html = default

        if validators is not None:
            self.validators = validators[:]

        # These are set up by `.bind()` when the field is added to a serializer.
        self.field_name = None
        self.parent = None

        # Collect default error message from self and parent classes
        messages = {}
        for cls in reversed(self.__class__.__mro__):
            messages.update(getattr(cls, 'default_error_messages', {}))
        messages.update(error_messages or {})
        self.error_messages = messages

    def bind(self, field_name, parent):
        """
        Initializes the field name and parent for the field instance.
        Called when a field is added to the parent serializer instance.
        """

        # In order to enforce a consistent style, we error if a redundant
        # 'source' argument has been used. For example:
        # my_field = serializer.CharField(source='my_field')
        assert self.source != field_name, (
            "It is redundant to specify `source='%s'` on field '%s' in "
            "serializer '%s', because it is the same as the field name. "
            "Remove the `source` keyword argument." %
            (field_name, self.__class__.__name__, parent.__class__.__name__)
        )

        self.field_name = field_name
        self.parent = parent

        # `self.label` should default to being based on the field name.
        if self.label is None:
            self.label = field_name.replace('_', ' ').capitalize()

        # self.source should default to being the same as the field name.
        if self.source is None:
            self.source = field_name

        # self.source_attrs is a list of attributes that need to be looked up
        # when serializing the instance, or populating the validated data.
        if self.source == '*':
            self.source_attrs = []
        else:
            self.source_attrs = self.source.split('.')

    # .validators is a lazily loaded property, that gets its default
    # value from `get_validators`.
    @property
    def validators(self):
        if not hasattr(self, '_validators'):
            self._validators = self.get_validators()
        return self._validators

    @validators.setter
    def validators(self, validators):
        self._validators = validators

    def get_validators(self):
        return self.default_validators[:]

    def get_initial(self):
        """
        Return a value to use when the field is being returned as a primitive
        value, without any object instance.
        """
        return self.initial

    def get_value(self, dictionary):
        """
        Given the *incoming* primitive data, return the value for this field
        that should be validated and transformed to a native value.
        """
        if html.is_html_input(dictionary):
            # HTML forms will represent empty fields as '', and cannot
            # represent None or False values directly.
            if self.field_name not in dictionary:
                if getattr(self.root, 'partial', False):
                    return empty
                return self.default_empty_html
            ret = dictionary[self.field_name]
            if ret == '' and self.allow_null:
                # If the field is blank, and null is a valid value then
                # determine if we should use null instead.
                return '' if getattr(self, 'allow_blank', False) else None
            elif ret == '' and not self.required:
                # If the field is blank, and emptyness is valid then
                # determine if we should use emptyness instead.
                return '' if getattr(self, 'allow_blank', False) else empty
            return ret
        return dictionary.get(self.field_name, empty)

    def get_attribute(self, instance):
        """
        Given the *outgoing* object instance, return the primitive value
        that should be used for this field.
        """
        try:
            return get_attribute(instance, self.source_attrs)
        except (KeyError, AttributeError) as exc:
            if not self.required and self.default is empty:
                raise SkipField()
            msg = (
                'Got {exc_type} when attempting to get a value for field '
                '`{field}` on serializer `{serializer}`.\nThe serializer '
                'field might be named incorrectly and not match '
                'any attribute or key on the `{instance}` instance.\n'
                'Original exception text was: {exc}.'.format(
                    exc_type=type(exc).__name__,
                    field=self.field_name,
                    serializer=self.parent.__class__.__name__,
                    instance=instance.__class__.__name__,
                    exc=exc
                )
            )
            raise type(exc)(msg)

    def get_default(self):
        """
        Return the default value to use when validating data if no input
        is provided for this field.
        If a default has not been set for this field then this will simply
        return `empty`, indicating that no value should be set in the
        validated data for this field.
        """
        if self.default is empty:
            raise SkipField()
        if callable(self.default):
            if hasattr(self.default, 'set_context'):
                self.default.set_context(self)
            return self.default()
        return self.default

    def validate_empty_values(self, data):
        """
        Validate empty values, and either:
        * Raise `ValidationError`, indicating invalid data.
        * Raise `SkipField`, indicating that the field should be ignored.
        * Return (True, data), indicating an empty value that should be
          returned without any further validation being applied.
        * Return (False, data), indicating a non-empty value, that should
          have validation applied as normal.
        """
        if self.read_only:
            return (True, self.get_default())

        if data is empty:
            if getattr(self.root, 'partial', False):
                raise SkipField()
            if self.required:
                self.fail('required')
            return (True, self.get_default())

        if data is None:
            if not self.allow_null:
                self.fail('null')
            return (True, None)

        return (False, data)

    def run_validation(self, data=empty):
        """
        Validate a simple representation and return the internal value.
        The provided data may be `empty` if no representation was included
        in the input.
        May raise `SkipField` if the field should not be included in the
        validated data.
        """
        (is_empty_value, data) = self.validate_empty_values(data)
        if is_empty_value:
            return data
        value = self.to_internal_value(data)
        self.run_validators(value)
        return value

    def run_validators(self, value):
        """
        Test the given value against all the validators on the field,
        and either raise a `ValidationError` or simply return.
        """
        errors = []
        for validator in self.validators:
            if hasattr(validator, 'set_context'):
                validator.set_context(self)

            try:
                validator(value)
            except ValidationError as exc:
                # If the validation error contains a mapping of fields to
                # errors then simply raise it immediately rather than
                # attempting to accumulate a list of errors.
                if isinstance(exc.detail, dict):
                    raise
                errors.extend(exc.detail)
            except DjangoValidationError as exc:
                errors.extend(exc.messages)
        if errors:
            raise ValidationError(errors)

    def to_internal_value(self, data):
        """
        Transform the *incoming* primitive data into a native value.
        """
        raise NotImplementedError(
            '{cls}.to_internal_value() must be implemented.'.format(
                cls=self.__class__.__name__
            )
        )

    def to_representation(self, value):
        """
        Transform the *outgoing* native value into primitive data.
        """
        raise NotImplementedError(
            '{cls}.to_representation() must be implemented for field '
            '{field_name}. If you do not need to support write operations '
            'you probably want to subclass `ReadOnlyField` instead.'.format(
                cls=self.__class__.__name__,
                field_name=self.field_name,
            )
        )

    def fail(self, key, **kwargs):
        """
        A helper method that simply raises a validation error.
        """
        try:
            msg = self.error_messages[key]
        except KeyError:
            class_name = self.__class__.__name__
            msg = MISSING_ERROR_MESSAGE.format(class_name=class_name, key=key)
            raise AssertionError(msg)
        message_string = msg.format(**kwargs)
        raise ValidationError(message_string)

    @cached_property
    def root(self):
        """
        Returns the top-level serializer for this field.
        """
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    @cached_property
    def context(self):
        """
        Returns the context as passed to the root serializer on initialization.
        """
        return getattr(self.root, '_context', {})

    def __new__(cls, *args, **kwargs):
        """
        When a field is instantiated, we store the arguments that were used,
        so that we can present a helpful representation of the object.
        """
        instance = super(Field, cls).__new__(cls)
        instance._args = args
        instance._kwargs = kwargs
        return instance

    def __deepcopy__(self, memo):
        """
        When cloning fields we instantiate using the arguments it was
        originally created with, rather than copying the complete state.
        """
        args = copy.deepcopy(self._args)
        kwargs = dict(self._kwargs)
        # Bit ugly, but we need to special case 'validators' as Django's
        # RegexValidator does not support deepcopy.
        # We treat validator callables as immutable objects.
        # See https://github.com/tomchristie/django-rest-framework/issues/1954
        validators = kwargs.pop('validators', None)
        kwargs = copy.deepcopy(kwargs)
        if validators is not None:
            kwargs['validators'] = validators
        return self.__class__(*args, **kwargs)

    def __repr__(self):
        """
        Fields are represented using their initial calling arguments.
        This allows us to create descriptive representations for serializer
        instances that show all the declared fields on the serializer.
        """
        return unicode_to_repr(representation.field_repr(self))


class IntegerField(Field):
    default_error_messages = {
        'invalid': _('A valid integer is required.'),
        'max_value': _('Ensure this value is less than or equal to {max_value}.'),
        'min_value': _('Ensure this value is greater than or equal to {min_value}.'),
        'max_string_length': _('String value too large.')
    }
    MAX_STRING_LENGTH = 1000  # Guard against malicious string inputs.
    re_decimal = re.compile(r'\.0*\s*$')  # allow e.g. '1.0' as an int, but not '1.2'

    def __init__(self, **kwargs):
        self.max_value = kwargs.pop('max_value', None)
        self.min_value = kwargs.pop('min_value', None)
        super(IntegerField, self).__init__(**kwargs)
        if self.max_value is not None:
            message = self.error_messages['max_value'].format(max_value=self.max_value)
            self.validators.append(MaxValueValidator(self.max_value, message=message))
        if self.min_value is not None:
            message = self.error_messages['min_value'].format(min_value=self.min_value)
            self.validators.append(MinValueValidator(self.min_value, message=message))

    def to_internal_value(self, data):
        if isinstance(data, six.text_type) and len(data) > self.MAX_STRING_LENGTH:
            self.fail('max_string_length')

        try:
            data = int(self.re_decimal.sub('', str(data)))
        except (ValueError, TypeError):
            self.fail('invalid')
        return data

    def to_representation(self, value):
        return int(value)


class HexIntegerField(IntegerField):
    """
    Store an integer represented as a hex string of form "0x01".
    """

    def to_internal_value(self, data):
        data = int(data, 16)
        return super(HexIntegerField, self).to_internal_value(data)

    def to_representation(self, value):
        return value


# Serializers
class DeviceSerializerMixin(ModelSerializer):
    class Meta:
        fields = ("name", "registration_id", "device_id", "active", "date_created")
        read_only_fields = ("date_created",)


class APNSDeviceSerializer(ModelSerializer):
    class Meta(DeviceSerializerMixin.Meta):
        model = APNSDevice

    def validate_registration_id(self, value, source):
        if hex_re.match(value[source]) is None or len(value[source]) != 64:
            raise ValidationError("Registration ID (device token) is invalid")
        return value


class GCMDeviceSerializer(ModelSerializer):
    device_id = HexIntegerField(
        help_text="ANDROID_ID / TelephonyManager.getDeviceId() (e.g: 0x01)",
        style={'input_type': 'text'},
        required=False
    )

    class Meta(DeviceSerializerMixin.Meta):
        model = GCMDevice


# Permissions
class IsOwner(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        # must be the owner to view the object
        return obj.user == request.user


# Mixins
class DeviceViewSetMixin(object):
    lookup_field = "registration_id"

    def perform_create(self, serializer):
        if self.request.user.is_authenticated():
            serializer.save(user=self.request.user)
        return super(DeviceViewSetMixin, self).perform_create(serializer)


class AuthorizedMixin(object):
    permission_classes = (permissions.IsAuthenticated, IsOwner)

    def get_queryset(self):
        # filter all devices to only those belonging to the current user
        return self.queryset.filter(user=self.request.user)


# ViewSets
class APNSDeviceViewSet(DeviceViewSetMixin, ModelViewSet):
    queryset = APNSDevice.objects.all()
    serializer_class = APNSDeviceSerializer


class APNSDeviceAuthorizedViewSet(AuthorizedMixin, APNSDeviceViewSet):
    pass


class GCMDeviceViewSet(DeviceViewSetMixin, ModelViewSet):
    queryset = GCMDevice.objects.all()
    serializer_class = GCMDeviceSerializer


class GCMDeviceAuthorizedViewSet(AuthorizedMixin, GCMDeviceViewSet):
    pass
