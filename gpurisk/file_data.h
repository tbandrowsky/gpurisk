#pragma once

/**********************************************************
oxsat - portable sat solver for windows and unix
Copyright (C) 2017  tj bandrowsky

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <string>
#include <iostream>
#include <fstream>
#include <clocale>
#include <locale>
#include "sys/stat.h"

namespace io
{
	class file_data
	{
		std::string file_name;
		std::string converted_str;
		int length;
		bool error;

	public:

		file_data() : error(false)
		{

		}

		bool get_error() const
		{
			return error;
		}

		/*		file_data(Platform::String^ str)
		{
		auto wdata = str->Data();
		using convert_type = std::codecvt_utf8<wchar_t>;
		std::wstring_convert<convert_type, wchar_t> converter;
		converted_str = converter.to_bytes(wdata);
		length = converted_str.size();
		}
		*/

		file_data(const char *cfilename) : error(false)
		{
			file_name = cfilename;

			struct stat stat_buf;
			int rc = stat(cfilename, &stat_buf);
			if (rc == 0)
			{
				length = stat_buf.st_size * 2 + 1;
				FILE *fp = fopen(cfilename, "r");
				if (fp != nullptr) {
					char *buffer = new char[length + 1];
					fread(buffer, 1, length, fp);
					fclose(fp);
					converted_str = buffer;
					delete[] buffer;
				}
				else
				{
					error = true;
				}
			}
			else {
				error = true;
			}
		}

		virtual ~file_data()
		{
		}

		const char *get_data()
		{
			return converted_str.c_str();
		}

		int get_data_length()
		{
			return length;
		}

		const std::string& get_file_name()
		{
			return file_name;
		}

	};

}
