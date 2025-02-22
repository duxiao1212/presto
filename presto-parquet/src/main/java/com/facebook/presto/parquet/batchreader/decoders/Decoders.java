/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.facebook.presto.parquet.batchreader.decoders;

import com.facebook.presto.parquet.DataPage;
import com.facebook.presto.parquet.DataPageV1;
import com.facebook.presto.parquet.DataPageV2;
import com.facebook.presto.parquet.ParquetEncoding;
import com.facebook.presto.parquet.RichColumnDescriptor;
import com.facebook.presto.parquet.batchreader.decoders.delta.BinaryDeltaValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.BinaryLongDecimalDeltaValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.BinaryShortDecimalDeltaValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.FixedLenByteArrayLongDecimalDeltaValueDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.FixedLenByteArrayShortDecimalDeltaValueDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.FixedLenByteArrayUuidDeltaValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.Int32DeltaBinaryPackedValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.Int32ShortDecimalDeltaValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.Int64DeltaBinaryPackedValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.Int64ShortDecimalDeltaValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.delta.Int64TimeAndTimestampMicrosDeltaBinaryPackedValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.BinaryLongDecimalPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.BinaryPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.BinaryShortDecimalPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.BooleanPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.FixedLenByteArrayLongDecimalPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.FixedLenByteArrayShortDecimalPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.FixedLenByteArrayUuidPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.Int32PlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.Int32ShortDecimalPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.Int64PlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.Int64ShortDecimalPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.Int64TimeAndTimestampMicrosPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.plain.TimestampPlainValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.BinaryRLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.BooleanRLEValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.Int32RLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.Int32ShortDecimalRLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.Int64RLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.Int64TimeAndTimestampMicrosRLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.LongDecimalRLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.ShortDecimalRLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.TimestampRLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.decoders.rle.UuidRLEDictionaryValuesDecoder;
import com.facebook.presto.parquet.batchreader.dictionary.BinaryBatchDictionary;
import com.facebook.presto.parquet.batchreader.dictionary.TimestampDictionary;
import com.facebook.presto.parquet.dictionary.Dictionary;
import com.facebook.presto.parquet.dictionary.IntegerDictionary;
import com.facebook.presto.parquet.dictionary.LongDictionary;
import com.facebook.presto.spi.PrestoException;
import io.airlift.slice.Slice;
import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.values.ValuesReader;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;

import static com.facebook.presto.parquet.ParquetEncoding.DELTA_BINARY_PACKED;
import static com.facebook.presto.parquet.ParquetEncoding.DELTA_BYTE_ARRAY;
import static com.facebook.presto.parquet.ParquetEncoding.DELTA_LENGTH_BYTE_ARRAY;
import static com.facebook.presto.parquet.ParquetEncoding.PLAIN;
import static com.facebook.presto.parquet.ParquetEncoding.PLAIN_DICTIONARY;
import static com.facebook.presto.parquet.ParquetEncoding.RLE;
import static com.facebook.presto.parquet.ParquetEncoding.RLE_DICTIONARY;
import static com.facebook.presto.parquet.ParquetErrorCode.PARQUET_IO_READ_ERROR;
import static com.facebook.presto.parquet.ParquetErrorCode.PARQUET_UNSUPPORTED_COLUMN_TYPE;
import static com.facebook.presto.parquet.ParquetErrorCode.PARQUET_UNSUPPORTED_ENCODING;
import static com.facebook.presto.parquet.ParquetTypeUtils.isDecimalType;
import static com.facebook.presto.parquet.ParquetTypeUtils.isShortDecimalType;
import static com.facebook.presto.parquet.ParquetTypeUtils.isTimeMicrosType;
import static com.facebook.presto.parquet.ParquetTypeUtils.isTimeStampMicrosType;
import static com.facebook.presto.parquet.ParquetTypeUtils.isUuidType;
import static com.facebook.presto.parquet.ValuesType.VALUES;
import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.String.format;
import static org.apache.parquet.bytes.BytesUtils.getWidthFromMaxInt;
import static org.apache.parquet.bytes.BytesUtils.readIntLittleEndian;
import static org.apache.parquet.bytes.BytesUtils.readIntLittleEndianOnOneByte;
import static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BOOLEAN;

public class Decoders
{
    private Decoders()
    {
    }

    public static FlatDecoders readFlatPage(DataPage page, RichColumnDescriptor columnDescriptor, Dictionary dictionary)
    {
        try {
            if (page instanceof DataPageV1) {
                return readFlatPageV1((DataPageV1) page, columnDescriptor, dictionary);
            }
            return readFlatPageV2((DataPageV2) page, columnDescriptor, dictionary);
        }
        catch (IOException e) {
            throw new PrestoException(PARQUET_IO_READ_ERROR, "Error reading parquet page " + page + " in column " + columnDescriptor, e);
        }
    }

    private static ValuesDecoder createValuesDecoder(ColumnDescriptor columnDescriptor, Dictionary dictionary, int valueCount, ParquetEncoding encoding, byte[] buffer, int offset, int length)
            throws IOException
    {
        final PrimitiveTypeName type = columnDescriptor.getPrimitiveType().getPrimitiveTypeName();

        if (encoding == PLAIN) {
            switch (type) {
                case BOOLEAN:
                    return new BooleanPlainValuesDecoder(buffer, offset, length);
                case INT32:
                    if (isShortDecimalType(columnDescriptor)) {
                        return new Int32ShortDecimalPlainValuesDecoder(buffer, offset, length);
                    }
                case FLOAT:
                    return new Int32PlainValuesDecoder(buffer, offset, length);
                case INT64: {
                    if (isTimeStampMicrosType(columnDescriptor)) {
                        LogicalTypeAnnotation.TimestampLogicalTypeAnnotation typeAnnotation = (LogicalTypeAnnotation.TimestampLogicalTypeAnnotation) columnDescriptor.getPrimitiveType().getLogicalTypeAnnotation();
                        boolean withTimezone = typeAnnotation.isAdjustedToUTC();
                        return new Int64TimeAndTimestampMicrosPlainValuesDecoder(buffer, offset, length, withTimezone);
                    }
                    if (isTimeMicrosType(columnDescriptor)) {
                        return new Int64TimeAndTimestampMicrosPlainValuesDecoder(buffer, offset, length);
                    }
                    if (isShortDecimalType(columnDescriptor)) {
                        return new Int64ShortDecimalPlainValuesDecoder(buffer, offset, length);
                    }
                }
                case DOUBLE:
                    return new Int64PlainValuesDecoder(buffer, offset, length);
                case INT96:
                    return new TimestampPlainValuesDecoder(buffer, offset, length);
                case BINARY:
                    if (isDecimalType(columnDescriptor)) {
                        if (isShortDecimalType(columnDescriptor)) {
                            return new BinaryShortDecimalPlainValuesDecoder(buffer, offset, length);
                        }
                        return new BinaryLongDecimalPlainValuesDecoder(buffer, offset, length);
                    }
                    return new BinaryPlainValuesDecoder(buffer, offset, length);
                case FIXED_LEN_BYTE_ARRAY:
                    if (isDecimalType(columnDescriptor)) {
                        if (isShortDecimalType(columnDescriptor)) {
                            return new FixedLenByteArrayShortDecimalPlainValuesDecoder(columnDescriptor, buffer, offset, length);
                        }

                        int typeLength = columnDescriptor.getPrimitiveType().getTypeLength();
                        return new FixedLenByteArrayLongDecimalPlainValuesDecoder(typeLength, buffer, offset, length);
                    }
                    else if (isUuidType(columnDescriptor)) {
                        int typeLength = columnDescriptor.getPrimitiveType().getTypeLength();
                        return new FixedLenByteArrayUuidPlainValuesDecoder(typeLength, buffer, offset, length);
                    }
                default:
                    throw new PrestoException(PARQUET_UNSUPPORTED_COLUMN_TYPE, format("Column: %s, Encoding: %s", columnDescriptor, encoding));
            }
        }

        if (encoding == RLE && type == BOOLEAN) {
            ByteBuffer byteBuffer = ByteBuffer.wrap(buffer, offset, length);
            byteBuffer.getInt(); // skip past the length
            return new BooleanRLEValuesDecoder(byteBuffer);
        }

        if (encoding == RLE_DICTIONARY || encoding == PLAIN_DICTIONARY) {
            InputStream inputStream = ByteBufferInputStream.wrap(ByteBuffer.wrap(buffer, offset, length));
            int bitWidth = readIntLittleEndianOnOneByte(inputStream);
            switch (type) {
                case INT32:
                case FLOAT: {
                    if (isShortDecimalType(columnDescriptor)) {
                        return new Int32ShortDecimalRLEDictionaryValuesDecoder(bitWidth, inputStream, (IntegerDictionary) dictionary);
                    }
                    return new Int32RLEDictionaryValuesDecoder(bitWidth, inputStream, (IntegerDictionary) dictionary);
                }
                case INT64: {
                    if (isTimeStampMicrosType(columnDescriptor)) {
                        LogicalTypeAnnotation.TimestampLogicalTypeAnnotation typeAnnotation = (LogicalTypeAnnotation.TimestampLogicalTypeAnnotation) columnDescriptor.getPrimitiveType().getLogicalTypeAnnotation();
                        boolean withTimezone = typeAnnotation.isAdjustedToUTC();
                        return new Int64TimeAndTimestampMicrosRLEDictionaryValuesDecoder(bitWidth, inputStream, (LongDictionary) dictionary, withTimezone);
                    }
                    if (isTimeMicrosType(columnDescriptor)) {
                        return new Int64TimeAndTimestampMicrosRLEDictionaryValuesDecoder(bitWidth, inputStream, (LongDictionary) dictionary, false);
                    }
                    if (isShortDecimalType(columnDescriptor)) {
                        return new Int64RLEDictionaryValuesDecoder(bitWidth, inputStream, (LongDictionary) dictionary);
                    }
                }
                case DOUBLE: {
                    return new Int64RLEDictionaryValuesDecoder(bitWidth, inputStream, (LongDictionary) dictionary);
                }
                case INT96: {
                    return new TimestampRLEDictionaryValuesDecoder(bitWidth, inputStream, (TimestampDictionary) dictionary);
                }
                case BINARY: {
                    return new BinaryRLEDictionaryValuesDecoder(bitWidth, inputStream, (BinaryBatchDictionary) dictionary);
                }
                case FIXED_LEN_BYTE_ARRAY:
                    if (isDecimalType(columnDescriptor)) {
                        if (isShortDecimalType(columnDescriptor)) {
                            return new ShortDecimalRLEDictionaryValuesDecoder(bitWidth, inputStream, (BinaryBatchDictionary) dictionary);
                        }
                        return new LongDecimalRLEDictionaryValuesDecoder(bitWidth, inputStream, (BinaryBatchDictionary) dictionary);
                    }
                    else if (isUuidType(columnDescriptor)) {
                        return new UuidRLEDictionaryValuesDecoder(bitWidth, inputStream, (BinaryBatchDictionary) dictionary);
                    }
                default:
                    throw new PrestoException(PARQUET_UNSUPPORTED_COLUMN_TYPE, format("Column: %s, Encoding: %s", columnDescriptor, encoding));
            }
        }

        if (encoding == DELTA_BINARY_PACKED) {
            ByteBufferInputStream inputStream = ByteBufferInputStream.wrap(ByteBuffer.wrap(buffer, offset, length));
            switch (type) {
                case INT32:
                    if (isShortDecimalType(columnDescriptor)) {
                        ValuesReader parquetReader = getParquetReader(encoding, columnDescriptor, valueCount, inputStream);
                        return new Int32ShortDecimalDeltaValuesDecoder(parquetReader);
                    }
                case FLOAT: {
                    return new Int32DeltaBinaryPackedValuesDecoder(valueCount, inputStream);
                }
                case INT64: {
                    if (isTimeStampMicrosType(columnDescriptor)) {
                        LogicalTypeAnnotation.TimestampLogicalTypeAnnotation typeAnnotation = (LogicalTypeAnnotation.TimestampLogicalTypeAnnotation) columnDescriptor.getPrimitiveType().getLogicalTypeAnnotation();
                        boolean withTimezone = typeAnnotation.isAdjustedToUTC();
                        return new Int64TimeAndTimestampMicrosDeltaBinaryPackedValuesDecoder(valueCount, inputStream, withTimezone);
                    }
                    if (isTimeMicrosType(columnDescriptor)) {
                        return new Int64TimeAndTimestampMicrosDeltaBinaryPackedValuesDecoder(valueCount, inputStream, false);
                    }
                    if (isShortDecimalType(columnDescriptor)) {
                        ValuesReader parquetReader = getParquetReader(encoding, columnDescriptor, valueCount, inputStream);
                        return new Int64ShortDecimalDeltaValuesDecoder(parquetReader);
                    }
                }
                case DOUBLE: {
                    return new Int64DeltaBinaryPackedValuesDecoder(valueCount, inputStream);
                }
                default:
                    throw new PrestoException(PARQUET_UNSUPPORTED_COLUMN_TYPE, format("Column: %s, Encoding: %s", columnDescriptor, encoding));
            }
        }

        if ((encoding == DELTA_BYTE_ARRAY || encoding == DELTA_LENGTH_BYTE_ARRAY) && type == PrimitiveTypeName.BINARY) {
            ByteBufferInputStream inputStream = ByteBufferInputStream.wrap(ByteBuffer.wrap(buffer, offset, length));
            if (isDecimalType(columnDescriptor)) {
                if (isShortDecimalType(columnDescriptor)) {
                    return new BinaryShortDecimalDeltaValuesDecoder(encoding, valueCount, inputStream);
                }

                return new BinaryLongDecimalDeltaValuesDecoder(encoding, valueCount, inputStream);
            }
            return new BinaryDeltaValuesDecoder(encoding, valueCount, inputStream);
        }

        if (encoding == DELTA_BYTE_ARRAY && type == PrimitiveTypeName.FIXED_LEN_BYTE_ARRAY) {
            if (isDecimalType(columnDescriptor)) {
                ByteBufferInputStream inputStream = ByteBufferInputStream.wrap(ByteBuffer.wrap(buffer, offset, length));
                ValuesReader parquetReader = getParquetReader(encoding, columnDescriptor, valueCount, inputStream);

                if (isShortDecimalType(columnDescriptor)) {
                    return new FixedLenByteArrayShortDecimalDeltaValueDecoder(parquetReader, columnDescriptor);
                }

                return new FixedLenByteArrayLongDecimalDeltaValueDecoder(parquetReader);
            }
            else if (isUuidType(columnDescriptor)) {
                ByteBufferInputStream inputStream = ByteBufferInputStream.wrap(ByteBuffer.wrap(buffer, offset, length));
                ValuesReader parquetReader = getParquetReader(encoding, columnDescriptor, valueCount, inputStream);
                return new FixedLenByteArrayUuidDeltaValuesDecoder(parquetReader);
            }
        }

        throw new PrestoException(PARQUET_UNSUPPORTED_ENCODING, format("Column: %s, Encoding: %s", columnDescriptor, encoding));
    }

    private static ValuesReader getParquetReader(ParquetEncoding encoding, ColumnDescriptor descriptor, int valueCount, ByteBufferInputStream inputStream)
            throws IOException
    {
        ValuesReader valuesReader = encoding.getValuesReader(descriptor, VALUES);
        valuesReader.initFromPage(valueCount, inputStream);
        return valuesReader;
    }

    private static FlatDecoders readFlatPageV1(DataPageV1 page, RichColumnDescriptor columnDescriptor, Dictionary dictionary)
            throws IOException
    {
        byte[] bytes = page.getSlice().getBytes();
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes, 0, bytes.length);
        FlatDefinitionLevelDecoder definitionLevelDecoder = createFlatDefinitionLevelDecoder(
                page.getDefinitionLevelEncoding(),
                columnDescriptor.isRequired(),
                columnDescriptor.getMaxDefinitionLevel(),
                page.getValueCount(),
                byteBuffer);
        ValuesDecoder valuesDecoder = createValuesDecoder(
                columnDescriptor,
                dictionary,
                page.getValueCount(),
                page.getValueEncoding(),
                bytes,
                byteBuffer.position(),
                bytes.length - byteBuffer.position());
        return new FlatDecoders(definitionLevelDecoder, valuesDecoder);
    }

    private static FlatDecoders readFlatPageV2(DataPageV2 pageV2, RichColumnDescriptor columnDescriptor, Dictionary dictionary)
            throws IOException
    {
        final int valueCount = pageV2.getValueCount();
        final int maxDefinitionLevel = columnDescriptor.getMaxDefinitionLevel();
        checkArgument(maxDefinitionLevel <= 1 && maxDefinitionLevel >= 0, "Invalid max definition level: " + maxDefinitionLevel);

        FlatDefinitionLevelDecoder definitionLevelDecoder;
        if (maxDefinitionLevel == 0) {
            definitionLevelDecoder = new FlatDefinitionLevelDecoder(0, valueCount);
        }
        else {
            definitionLevelDecoder = new FlatDefinitionLevelDecoder(valueCount, new ByteArrayInputStream(pageV2.getDefinitionLevels().getBytes()));
        }
        ValuesDecoder valuesDecoder = createValuesDecoderV2(pageV2, columnDescriptor, dictionary);
        return new FlatDecoders(definitionLevelDecoder, valuesDecoder);
    }

    public static NestedDecoders readNestedPage(DataPage page, RichColumnDescriptor columnDescriptor, Dictionary dictionary)
    {
        try {
            if (page instanceof DataPageV1) {
                return readNestedPageV1((DataPageV1) page, columnDescriptor, dictionary);
            }
            return readNestedPageV2((DataPageV2) page, columnDescriptor, dictionary);
        }
        catch (IOException e) {
            throw new PrestoException(PARQUET_IO_READ_ERROR, "Error reading parquet page " + page + " in column " + columnDescriptor, e);
        }
    }

    private static NestedDecoders readNestedPageV1(DataPageV1 page, RichColumnDescriptor columnDescriptor, Dictionary dictionary)
            throws IOException
    {
        byte[] bytes = page.getSlice().getBytes();
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes, 0, bytes.length);

        RepetitionLevelDecoder repetitionLevelDecoder = createRepetitionLevelDecoder(
                page.getRepetitionLevelEncoding(),
                columnDescriptor.getMaxRepetitionLevel(),
                page.getValueCount(),
                byteBuffer);
        DefinitionLevelDecoder definitionLevelDecoder = createDefinitionLevelDecoder(
                page.getDefinitionLevelEncoding(),
                columnDescriptor.getMaxDefinitionLevel(),
                page.getValueCount(),
                byteBuffer);
        ValuesDecoder valuesDecoder = createValuesDecoder(
                columnDescriptor,
                dictionary,
                page.getValueCount(),
                page.getValueEncoding(),
                bytes,
                byteBuffer.position(),
                bytes.length - byteBuffer.position());
        return new NestedDecoders(repetitionLevelDecoder, definitionLevelDecoder, valuesDecoder);
    }

    private static NestedDecoders readNestedPageV2(DataPageV2 pageV2, RichColumnDescriptor columnDescriptor, Dictionary dictionary)
            throws IOException
    {
        final int valueCount = pageV2.getValueCount();
        return new NestedDecoders(
                createRepetitionLevelDecoderV2(valueCount, columnDescriptor, pageV2.getRepetitionLevels()),
                createDefinitionLevelDecoderV2(valueCount, columnDescriptor, pageV2.getDefinitionLevels()),
                createValuesDecoderV2(pageV2, columnDescriptor, dictionary));
    }

    private static RepetitionLevelDecoder createRepetitionLevelDecoderV2(int valueCount, RichColumnDescriptor columnDescriptor, Slice repetitionLevelBuffer)
    {
        final int maxRepetitionLevel = columnDescriptor.getMaxRepetitionLevel();
        final int repetitionLevelBitWidth = getWidthFromMaxInt(maxRepetitionLevel);
        if (maxRepetitionLevel == 0 || repetitionLevelBitWidth == 0) {
            return new RepetitionLevelDecoder(0, valueCount);
        }
        return new RepetitionLevelDecoder(valueCount, repetitionLevelBitWidth, new ByteArrayInputStream(repetitionLevelBuffer.getBytes()));
    }

    private static DefinitionLevelDecoder createDefinitionLevelDecoderV2(int valueCount, RichColumnDescriptor columnDescriptor, Slice definitionLevelBuffer)
    {
        final int maxDefinitionLevel = columnDescriptor.getMaxDefinitionLevel();
        final int definitionLevelBitWidth = getWidthFromMaxInt(maxDefinitionLevel);
        if (maxDefinitionLevel == 0 || definitionLevelBitWidth == 0) {
            return new DefinitionLevelDecoder(0, valueCount);
        }
        return new DefinitionLevelDecoder(valueCount, definitionLevelBitWidth, new ByteArrayInputStream(definitionLevelBuffer.getBytes()));
    }

    private static ValuesDecoder createValuesDecoderV2(DataPageV2 pageV2, RichColumnDescriptor columnDescriptor, Dictionary dictionary)
            throws IOException
    {
        final byte[] valueBuffer = pageV2.getSlice().getBytes();
        return createValuesDecoder(columnDescriptor, dictionary, pageV2.getValueCount(), pageV2.getDataEncoding(), valueBuffer, 0, valueBuffer.length);
    }

    private static FlatDefinitionLevelDecoder createFlatDefinitionLevelDecoder(ParquetEncoding encoding, boolean isRequired, int maxLevelValue, int valueCount, ByteBuffer buffer)
            throws IOException
    {
        if (isRequired) {
            return new FlatDefinitionLevelDecoder(1, valueCount);
        }

        if (maxLevelValue == 0) {
            return new FlatDefinitionLevelDecoder(0, valueCount);
        }

        final int bitWidth = getWidthFromMaxInt(maxLevelValue);
        if (bitWidth == 0) {
            return new FlatDefinitionLevelDecoder(0, valueCount);
        }

        checkArgument(encoding == RLE, "Invalid definition level encoding: " + encoding);
        ByteBufferInputStream bufferInputStream = ByteBufferInputStream.wrap(buffer);

        final int bufferSize = readIntLittleEndian(bufferInputStream);
        FlatDefinitionLevelDecoder definitionLevelDecoder = new FlatDefinitionLevelDecoder(valueCount, bufferInputStream.sliceStream(bufferSize));
        ((Buffer) buffer).position(buffer.position() + bufferSize + 4);
        return definitionLevelDecoder;
    }

    public static RepetitionLevelDecoder createRepetitionLevelDecoder(ParquetEncoding encoding, int maxLevelValue, int valueCount, ByteBuffer buffer)
            throws IOException
    {
        final int bitWidth = getWidthFromMaxInt(maxLevelValue);
        if (maxLevelValue == 0 || bitWidth == 0) {
            return new RepetitionLevelDecoder(0, valueCount);
        }

        checkArgument(encoding == RLE, "Invalid repetition level encoding: " + encoding);
        ByteBufferInputStream bufferInputStream = ByteBufferInputStream.wrap(buffer);

        final int bufferSize = readIntLittleEndian(bufferInputStream);
        RepetitionLevelDecoder repetitionLevelDecoder = new RepetitionLevelDecoder(valueCount, bitWidth, bufferInputStream.sliceStream(bufferSize));
        ((Buffer) buffer).position(buffer.position() + bufferSize + 4);
        return repetitionLevelDecoder;
    }

    public static DefinitionLevelDecoder createDefinitionLevelDecoder(ParquetEncoding encoding, int maxLevelValue, int valueCount, ByteBuffer buffer)
            throws IOException
    {
        final int bitWidth = getWidthFromMaxInt(maxLevelValue);
        if (maxLevelValue == 0 || bitWidth == 0) {
            return new DefinitionLevelDecoder(0, valueCount);
        }

        checkArgument(encoding == RLE, "Invalid definition level encoding: " + encoding);
        ByteBufferInputStream bufferInputStream = ByteBufferInputStream.wrap(buffer);

        final int bufferSize = readIntLittleEndian(bufferInputStream);
        DefinitionLevelDecoder definitionLevelDecoder = new DefinitionLevelDecoder(valueCount, bitWidth, bufferInputStream.sliceStream(bufferSize));
        ((Buffer) buffer).position(buffer.position() + bufferSize + 4);
        return definitionLevelDecoder;
    }

    public static class FlatDecoders
    {
        private final FlatDefinitionLevelDecoder definitionLevelDecoder;
        private final ValuesDecoder valuesDecoder;

        private FlatDecoders(FlatDefinitionLevelDecoder dlDecoder, ValuesDecoder valuesDecoder)
        {
            this.definitionLevelDecoder = dlDecoder;
            this.valuesDecoder = valuesDecoder;
        }

        public FlatDefinitionLevelDecoder getDefinitionLevelDecoder()
        {
            return definitionLevelDecoder;
        }

        public ValuesDecoder getValuesDecoder()
        {
            return valuesDecoder;
        }
    }

    public static class NestedDecoders
    {
        private final RepetitionLevelDecoder repetitionLevelDecoder;
        private final DefinitionLevelDecoder definitionLevelDecoder;
        private final ValuesDecoder valuesDecoder;

        private NestedDecoders(RepetitionLevelDecoder repetitionLevelDecoder, DefinitionLevelDecoder definitionLevelDecoder, ValuesDecoder valuesDecoder)
        {
            this.repetitionLevelDecoder = repetitionLevelDecoder;
            this.definitionLevelDecoder = definitionLevelDecoder;
            this.valuesDecoder = valuesDecoder;
        }

        public RepetitionLevelDecoder getRepetitionLevelDecoder()
        {
            return repetitionLevelDecoder;
        }

        public DefinitionLevelDecoder getDefinitionLevelDecoder()
        {
            return definitionLevelDecoder;
        }

        public ValuesDecoder getValuesDecoder()
        {
            return valuesDecoder;
        }
    }
}
